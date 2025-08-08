from .entities import (
    dt2str,
    str2dt,
    User,
    UserJSON,
    Station,
    StationJSON,
    StationStatus,
    StationStatusJSON,
    Reading,
    Location,
    DeviceType,
    CameraModel,
    UserRole,
)
from .DBManager import Manager
from pydantic import BaseModel, EmailStr, SecretStr
from pydantic_extra_types.mac_address import MacAddress
from pydantic_extra_types.coordinate import Latitude, Longitude
from sqlite3 import Error as SQLError
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Union
from uuid import uuid4
from logging import ERROR, DEBUG
from datetime import datetime, UTC

T = TypeVar("T", bound=BaseModel)


logcursorfailure = lambda: Manager.log("Failed to get cursor.", level=ERROR)


class DatabaseError(Exception):
    """
    Custom exception for database-related errors.
    """

    def __init__(self, message: str):
        super().__init__(message)
        Manager.log(message, level=ERROR)


class NotFoundError(DatabaseError): ...


class AlreadyExistsError(DatabaseError): ...


class InvalidInputError(DatabaseError): ...


class InternalDBError(DatabaseError): ...


class UnauthorizedError(DatabaseError): ...


class Service(ABC, Generic[T]):
    """
    Abstract class for a services, in-between class for db transactions.

    Methods:
        - get: Get an entity from the database.
        - list: Get a slice of entities from the database.
        - insert: Add an entity to the database.
        - update: Update an entity in the database.
        - delete: Delete an entity from the database.
        - exists: Check if an entity exists in the database.
    """

    @staticmethod
    @abstractmethod
    def get(**kwargs) -> Optional[T]:
        """
        Get all entities from the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def list(**kwargs) -> list[T]:
        """
        Get a slice of Entities from the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def insert(**kwargs) -> None:
        """
        Add an entity to the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def update(**kwargs) -> None:
        """
        Update an entity in the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def delete(**kwargs) -> None:
        """
        Delete an entity from the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def exists(**kwargs) -> bool:
        """
        Check if an entity exists in the database.
        """
        pass


class UserService(Service[User]):

    @staticmethod
    def get(**kwargs) -> Optional[User]:
        """
        Get a user from the database.
        Args:
            - email (str): The email of the user to retrieve.
            - id (str): The ID of the user to retrieve.

        Returns:
            - Optional[User]: The user object if found, None otherwise.

        Raises:
            InvalidInputError: If neither email nor id is provided.
            InternalDBError: If there is an error getting the cursor for user retrieval.
            SQLError: If there is an error executing the SQL query.
        """

        result: Optional[User] = None
        email: Optional[EmailStr] = kwargs.get("email", None)
        id: Optional[str] = kwargs.get("id", None)

        if not email and not id:
            raise InvalidInputError(
                f"Either 'email' or 'id' must be provided to retrieve a user."
            )

        if email:
            queryparam = "email = ?;"
            params = (email,)

        if id:
            queryparam = f"ID = ?;"
            params = (id,)

        query = f"SELECT * FROM users WHERE {queryparam}"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for user retrieval.")

            cursor.execute(query, params)
            data = cursor.fetchone()
            if data:
                result = User(**data)

        return result

    @staticmethod
    def list(**kwargs) -> list[User]:
        """
        Get a slice of users from the database.
        Args:
            limit (int): The maximum number of users to retrieve.
            page (int): The page number to retrieve (0-indexed).
        Returns:
            list[User]: A list of user objects.
        Raises:
            InternalDBError: If there is an error getting the cursor for user listing.
            SQLError: If there is an error executing the SQL query.
        """
        results: list[User] = []
        limit: int = kwargs.get("limit", 20)
        page: int = kwargs.get("page", 0)
        offset = page * limit
        query = "SELECT * FROM users ORDER BY name LIMIT ? OFFSET ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                logcursorfailure()
                raise InternalDBError("Failed to get cursor for user listing.")

            cursor.execute(query, (limit, offset))
            data = cursor.fetchall()
            for row in data:
                results.append(
                    User(
                        ID=row[0],
                        name=row[1],
                        email=row[2],
                        password=row[3],
                        role=UserRole.match(row[4]),
                    )
                )

        return results

    @staticmethod
    def insert(**kwargs) -> None:
        """
        Insert a user into the database.
        Args:
            user (User): The user to insert.
        Raises:
            InvalidInputError: If no user is provided | if the provided user is not an instance of User.
            InternalDBError: If there is an error getting the cursor for user insertion.
            SQLError: If there is an error executing the SQL query.
        """
        user: Optional[User] = kwargs.get("user", None)
        if not user:
            Manager.log("No user provided for insertion.", level=ERROR)
            raise InvalidInputError("No user provided for insertion.")

        if not isinstance(user, User):
            Manager.log("Provided user is not an instance of User.", level=ERROR)
            raise InvalidInputError("Provided user is not an instance of User.")

        query = "INSERT INTO Users VALUES (?, ?, ?, ?, ?);"
        Manager.log(f"Inserting user :: {user}", level=DEBUG)

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for user insertion.")

            cursor.execute(
                query,
                (
                    user.ID,
                    user.name,
                    user.email,
                    Manager.hash_password(password=user.password.get_secret_value()),
                    user.role.value,
                ),
            )

    @staticmethod
    def update(**kwargs) -> None:
        """
        Update a user in the database.
        Args:
            id (str): The ID of the user to update.
            email (EmailStr): The email of the user to update.
            user (UserJSON): The user data to update.
        Raises:
            InvalidInputError: If neither id nor email is provided | if no user data is provided
            InternalDBError: If there is an error getting the cursor for user update.
            SQLError: If there is an error executing the SQL query.
            NotFoundError: If no user is found to update.
        """
        id: Optional[str] = kwargs.get("id", None)
        email: Optional[EmailStr] = kwargs.get("email", None)
        user: Optional[UserJSON] = kwargs.get("user", None)

        if not id and not email:
            raise InvalidInputError(
                "Either 'id' or 'email' must be provided to update a user."
            )

        if not user:
            raise InvalidInputError("No user provided for update.")

        params = []

        userparam = ""
        for key, value in user.items():
            if value is not None:
                userparam += f"{key} = ?, "

                if isinstance(value, UserRole):
                    params.append(value.value)
                    continue

                if isinstance(value, SecretStr):
                    params.append(Manager.hash_password(value.get_secret_value()))
                    continue

                params.append(value)

        if email:
            queryparam = "email = ?"
            params = [email]

        if id:
            queryparam = "ID = ?"
            params = [id]

        if not userparam:
            raise InvalidInputError("No fields to update in user.")

        userparam = userparam.rstrip(", ")

        query = f"UPDATE Users SET {userparam} WHERE {queryparam};"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for user update.")

            paramstup = tuple(params)

            cursor.execute(query, paramstup)

            if cursor.rowcount == 0:
                raise NotFoundError("No user found to update.")

    @staticmethod
    def delete(**kwargs) -> None:
        """
        Delete a user from the database.
        Args:
            id (str): The ID of the user to delete.
            email (EmailStr): The email of the user to delete.
        Raises:
            InvalidInputError: If neither id nor email is provided.
            InternalDBError: If there is an error getting the cursor for user deletion.
            SQLError: If there is an error executing the SQL query.
            NotFoundError: If no user is found to delete.
        """
        id: Optional[str] = kwargs.get("id", None)
        email: Optional[EmailStr] = kwargs.get("email", None)

        if not id and not email:
            raise InvalidInputError(
                "Either 'id' or 'email' must be provided to delete a user."
            )

        if email:
            queryparam = "email = ?"
            params = (email,)

        if id:
            queryparam = "ID = ?"
            params = (id,)

        query = f"DELETE FROM Users WHERE {queryparam};"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for user deletion.")

            cursor.execute(query, params)

            if cursor.rowcount == 0:
                raise NotFoundError("No user found to delete.")


class StatusService(Service[StationStatus]):
    """
    Service class for managing StationStatus entities in the database.
    """

    @staticmethod
    def get(**kwargs) -> Optional[StationStatus]:
        """
        Get a station status from the database.
        Args:
            MAC (MacAddress): The MAC address of the station to retrieve.

        Returns:
            Optional[StationStatus]: The station status object if found, None otherwise.

        Raises:
            InvalidInputError: If MAC is not provided | if there is an error retrieving the station status.
            InternalDBError: If there is an error getting the cursor for station status retrieval.
            SQLError: If there is an error executing the SQL query.
        """
        result: Optional[StationStatus] = None
        mac: Optional[MacAddress] = kwargs.get("MAC", None)

        if not mac:
            Manager.log(
                f"MAC must be provided to retrieve a station status.", level=ERROR
            )
            raise InvalidInputError(
                f"MAC must be provided to retrieve a station status."
            )

        query = "SELECT * FROM Status WHERE MAC = ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError(
                    "Failed to get cursor for station status retrieval."
                )

            cursor.execute(query, (mac,))
            data = cursor.fetchone()
            if data:
                result = StationStatus(
                    MAC=data[0],
                    timestamp=str2dt(data[1]),
                    SHT=data[2],
                    BMP=data[3],
                    CAM=data[4],
                    WIFI=data[5],
                )

        return result

    @staticmethod
    def insert(**kwargs) -> None:
        """
        Insert a station status into the database.
        Args:
            status (StationStatus): The station status to insert.
        Raises:
            InvalidInputError: If no station status is provided | if the provided station status is not an instance of StationStatus.
            InternalDBError: If there is an error getting the cursor for station status insertion.
            SQLError: If there is an error executing the SQL query.
        """
        status: Optional[StationStatus] = kwargs.get("status", None)
        if not status:
            raise InvalidInputError("No station status provided for insertion.")

        if not isinstance(status, StationStatus):
            raise InvalidInputError(
                "Provided station status is not an instance of StationStatus."
            )

        query = "INSERT INTO Status VALUES (?, ?, ?, ?, ?, ?);"
        Manager.log(f"Inserting station status :: {status}", level=DEBUG)

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError(
                    "Failed to get cursor for station status insertion."
                )

            cursor.execute(
                query,
                (
                    status.MAC,
                    dt2str(status.timestamp),
                    int(status.SHT),
                    int(status.BMP),
                    int(status.CAM),
                    int(status.WIFI),
                ),
            )

    @staticmethod
    def list(**kwargs) -> list[StationStatus]:
        """
        List all station statuses from the database.
        Args:
            limit (int): The maximum number of station statuses to retrieve.
            page (int): The page number to retrieve (0-indexed).
        Returns:
            list[StationStatus]: A list of station status objects.
        Raises:
            InternalDBError: If there is an error getting the cursor for station status listing.
            SQLError: If there is an error executing the SQL query.
        """
        results: list[StationStatus] = []
        limit: int = kwargs.get("limit", 20)
        page: int = kwargs.get("page", 0)
        offset = page * limit
        query = "SELECT * FROM Status ORDER BY timestamp LIMIT ? OFFSET ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                logcursorfailure()
                raise InternalDBError(
                    "Failed to get cursor for station status listing."
                )

            cursor.execute(query, (limit, offset))
            data = cursor.fetchall()
            for row in data:
                results.append(
                    StationStatus(
                        MAC=row[0],
                        timestamp=row[1],
                        SHT=bool(row[2]),
                        BMP=bool(row[3]),
                        CAM=bool(row[4]),
                        WIFI=bool(row[5]),
                    )
                )

        return results

    @staticmethod
    def update(**kwargs) -> None:
        """
        Update a station status in the database.
        Args:
            MAC (MacAddress): The MAC address of the station to update.
            status (StationStatus): The station status data to update.
        Raises:
            InvalidInputError: If MAC is not provided | if no station status data is provided | if the provided station status data is not an instance of StationStatusJSON.
            InternalDBError: If there is an error getting the cursor for station status update.
            SQLError: If there is an error executing the SQL query.
            NotFoundError: If no station status is found to update.
        """
        mac: Optional[MacAddress] = kwargs.get("MAC", None)
        status: Optional[StationStatus] = kwargs.get("status", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to update a station status.")

        if not status:
            raise InvalidInputError("No station status provided for update.")

        if not isinstance(status, StationStatus):
            raise InvalidInputError(
                "Provided station status is not an instance of StationStatus."
            )

        query = f"UPDATE Status SET timestamp = ?, SHT = ?, BMP = ?, CAM = ?, WIFI = ? WHERE MAC = ?;"
        params = (
            status.timestamp,
            status.SHT,
            status.BMP,
            status.CAM,
            status.WIFI,
            status.MAC,
        )

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for station status update.")

            cursor.execute(query, params)

            if cursor.rowcount == 0:
                raise NotFoundError("No station status found to update.")

    @staticmethod
    def delete(**kwargs) -> None:
        """
        Delete a station status from the database.
        Args:
            MAC (MacAddress): The MAC address of the station status to delete.
        Raises:
            InvalidInputError: If MAC is not provided.
            InternalDBError: If there is an error getting the cursor for station status deletion.
            SQLError: If there is an error executing the SQL query.
            NotFoundError: If no station status is found to delete.
        """
        mac: Optional[MacAddress] = kwargs.get("MAC", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to delete a station status.")

        query = "DELETE FROM Status WHERE MAC = ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                logcursorfailure()
                raise InternalDBError(
                    "Failed to get cursor for station status deletion."
                )

            cursor.execute(query, (mac,))

            if cursor.rowcount == 0:
                Manager.log("No station status found to delete.", level=ERROR)
                raise NotFoundError("No station status found to delete.")


class StationService(Service[Station]):
    """
    Service class for managing Station entities in the database.
    """

    @staticmethod
    def get(**kwargs) -> Optional[Station]:
        """
        Get a station from the database.
        Args:
            - MAC (MacAddress): The MAC address of the station to retrieve.

        Returns:
            - Optional[Station]: The station object if found, None otherwise.

        Raises:
            InvalidInputError: If MAC is not provided | if there is an error retrieving the station.
            InternalDBError: If there is an error getting the cursor for station retrieval.
            SQLError: If there is an error executing the SQL query.
        """
        result: Optional[Station] = None
        mac: Optional[MacAddress] = kwargs.get("MAC", None)

        if not mac:
            raise InvalidInputError(
                f"Either MAC or must be provided to retrieve a station."
            )

        query = f"SELECT * FROM Stations WHERE MAC = ? LIMIT 1;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for station retrieval.")

            cursor.execute(query, (mac,))
            data = cursor.fetchone()
            if data:

                status = StatusService.get(MAC=mac)

                result = Station(
                    MAC=mac,
                    name=data[1],
                    device_model=DeviceType[data[2]],
                    camera_model=CameraModel[data[3]],
                    firmware_version=data[4],
                    altitude=data[5],
                    latitude=Latitude(data[6]),
                    longitude=Longitude(data[7]),
                    sensors=status,
                )

        return result

    @staticmethod
    def list(**kwargs) -> list[Station]:
        """
        Get a slice of stations from the database.
        Args:
            limit (int): The maximum number of stations to retrieve.
            page (int): The page number to retrieve (0-indexed).
        Returns:
            list[Station]: A list of station objects.
        Raises:
            InternalDBError: If there is an error getting the cursor for station listing.
            SQLError: If there is an error executing the SQL query.
        """
        results: list[Station] = []
        limit: int = kwargs.get("limit", 20)
        page: int = kwargs.get("page", 0)
        offset = page * limit
        query = "SELECT * FROM Stations ORDER BY name LIMIT ? OFFSET ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for station listing.")

            cursor.execute(query, (limit, offset))
            data = cursor.fetchall()
            for row in data:
                mac = row[0]
                status = StatusService.get(MAC=mac)
                results.append(
                    Station(
                        MAC=mac,
                        name=row[1],
                        device_model=DeviceType.match(row[2]),
                        camera_model=CameraModel.match(row[3]),
                        firmware_version=row[4],
                        altitude=row[5],
                        latitude=Latitude(row[6]),
                        longitude=Longitude(row[7]),
                        sensors=status,
                    )
                )

        return results

    @staticmethod
    def insert(**kwargs) -> None:
        """
        Insert a station into the database.
        Args:
            station (Station): The station to insert.
        Raises:
            InvalidInputError: If no station is provided | if the provided station is not an instance of Station.
            InternalDBError: If there is an error getting the cursor for station insertion.
            SQLError: If there is an error executing the SQL query.
        """
        station: Optional[Station] = kwargs.get("station", None)
        if not station:
            raise InvalidInputError("No station provided for insertion.")

        if not isinstance(station, Station):
            raise InvalidInputError("Provided station is not an instance of Station.")

        query = "INSERT INTO Stations VALUES (?, ?, ?, ?, ?, ?, ?, ?);"
        Manager.log(f"Inserting station :: {station}", level=DEBUG)

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for station insertion.")

            cursor.execute(
                query,
                (
                    station.MAC,
                    station.name,
                    station.device_model.value,
                    station.camera_model.value,
                    station.firmware_version,
                    station.altitude,
                    float(station.latitude),
                    float(station.longitude),
                ),
            )

            station.sensors = StationStatus(
                MAC=station.MAC,
                timestamp=datetime.now(tz=UTC),
                SHT=False,
                BMP=False,
                CAM=False,
                WIFI=False,
            )
        StatusService.insert(status=station.sensors)

    @staticmethod
    def update(**kwargs) -> None:
        """
        Update a station in the database.
        Args:
            MAC (MacAddress): The MAC address of the station to update.
            station (StationJSON): The station data to update.
        Raises:
            InvalidInputError: If MAC is not provided | if no station data is provided | if the provided station data is not an instance of StationJSON.
            InternalDBError: If there is an error getting the cursor for station update.
            SQLError: If there is an error executing the SQL query.
            NotFoundError: If no station is found to update.
        """
        mac: Optional[MacAddress] = kwargs.get("MAC", None)
        station: Optional[StationJSON] = kwargs.get("station", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to update a station.")

        if not station:
            raise InvalidInputError("No station provided for update.")

        station.pop("sensors", None)
        params = []
        stationparam = ""

        for key, value in station.items():
            if value is not None:
                stationparam += f"{key} = ?, "

                if isinstance(value, (CameraModel, DeviceType)):
                    params.append(value.value)
                    continue

                params.append(value)

        if not stationparam:
            raise InvalidInputError("No fields to update in station.")

        stationparam = stationparam.rstrip(", ")
        params.append(mac)
        query = f"UPDATE Stations SET {stationparam} WHERE MAC = ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for station update.")

            paramstup = tuple(params)
            cursor.execute(query, paramstup)

            if cursor.rowcount == 0:
                raise NotFoundError("No station found to update.")

    @staticmethod
    def delete(**kwargs) -> None:
        """
        Delete a station from the database.
        Args:
            MAC (MacAddress): The MAC address of the station to delete.
        Raises:
            InvalidInputError: If MAC is not provided.
            InternalDBError: If there is an error getting the cursor for station deletion.
            SQLError: If there is an error executing the SQL query.
            NotFoundError: If no station is found to delete.
        """
        mac: Optional[MacAddress] = kwargs.get("MAC", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to delete a station.")

        query = "DELETE FROM Stations WHERE MAC = ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for station deletion.")

            cursor.execute(query, (mac,))

            if cursor.rowcount == 0:
                raise NotFoundError("No station found to delete.")

        StatusService.delete(MAC=mac)

    @staticmethod
    def exists(**kwargs) -> bool:
        """
        Check if a station exists in the database.
        Args:
            MAC (MacAddress): The MAC address of the station to check.
        Returns:
            bool: True if the station exists, False otherwise.
        Raises:
            InvalidInputError: If MAC is not provided.
            InternalDBError: If there is an error getting the cursor for station existence check.
            SQLError: If there is an error executing the SQL query.
        """
        mac: Optional[MacAddress] = kwargs.get("MAC", None)

        if not mac:
            raise InvalidInputError(
                "MAC must be provided to check if a station exists."
            )

        query = "SELECT 1 FROM Stations WHERE MAC = ? LIMIT 1;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError(
                    "Failed to get cursor for station existence check."
                )

            cursor.execute(query, (mac,))
            return cursor.fetchone() is not None
