from .entities import (
    User,
    UserJson,
    Station,
    StationStatus,
    Reading,
    Location,
    Entity,
    DeviceType,
    CameraModel,
    UserRole,
)
from .DBManager import Manager
from pydantic import BaseModel, EmailStr
from pydantic_extra_types.mac_address import MacAddress
from pydantic_extra_types.coordinate import Latitude, Longitude
from sqlite3 import Error as SQLError
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, Union
from uuid import uuid4
from logging import ERROR, DEBUG

T = TypeVar("T", bound=BaseModel)


logcursorfailure = lambda: Manager.log("Failed to get cursor.", level=ERROR)


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
            - ValueError: If neither email nor id is provided | if there is an error retrieving the user.
        """

        result: Optional[User] = None
        email: Optional[EmailStr] = kwargs.get("email", None)
        id: Optional[str] = kwargs.get("id", None)

        if not email and not id:
            Manager.log(f"Either 'email' or 'id' must be provided to retrieve a user.", level=ERROR)
            raise ValueError(f"Either 'email' or 'id' must be provided to retrieve a user.")    

        if email:
            queryparam = "email = ?;"
            params = (email,)

        if id:
            queryparam = f"ID = ?;"
            params = (id,)

        query = f"SELECT * FROM users WHERE {queryparam}"

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise ValueError("Failed to get cursor for user retrieval.")

                cursor.execute(query, params)
                data = cursor.fetchone()
                if data:
                    result = User(**data)

        except SQLError as err:
            Manager.log(f"Error retrieving user :: {err}", level=ERROR)
            raise ValueError(f"Error retrieving user :: {err}")

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
        """
        results: list[User] = []
        limit: int = kwargs.get("limit", 20)
        page: int = kwargs.get("page", 0)
        offset = page * limit
        query = "SELECT * FROM users ORDER BY name LIMIT ? OFFSET ?;"

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise ValueError("Failed to get cursor for user listing.")

                cursor.execute(query, (limit, offset))
                data = cursor.fetchall()
                for row in data:
                    results.append(User(
                        ID=row[0],
                        name=row[1],
                        email=row[2],
                        password=row[3],
                        role=UserRole.match(row[4]),
                    ))

        except SQLError as err:
            Manager.log(f"Error listing users: {err}", level=ERROR)
            raise ValueError(f"Error listing users: {err}")

        return results

    @staticmethod
    def insert(**kwargs) -> None:
        user: Optional[User] = kwargs.get("user", None)
        if not user:
            Manager.log("No user provided for insertion.", level=ERROR)
            raise ValueError("No user provided for insertion.")
        
        if not isinstance(user, User):
            Manager.log("Provided user is not an instance of User.", level=ERROR)
            raise ValueError("Provided user is not an instance of User.")
        
        query = "INSERT INTO Users VALUES (?, ?, ?, ?, ?);"
        Manager.log(f"Inserting user :: {user}", level=DEBUG)

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise ValueError("Failed to get cursor for user insertion.")

                cursor.execute(query, (
                    user.ID,
                    user.name,
                    user.email,
                    Manager.hash_password(user.password),
                    user.role.value,
                ))
            
        except SQLError as err:
            Manager.log(f"Error inserting user: {err}", level=ERROR)
            raise ValueError(f"Error inserting user: {err}")

    @staticmethod
    def update(**kwargs) -> None:
        id: Optional[str] = kwargs.get("id", None)
        email: Optional[EmailStr] = kwargs.get("email", None)
        user: Optional[UserJson] = kwargs.get("user", None)

        if not id and not email:
            Manager.log("Either 'id' or 'email' must be provided to update a user.", level=ERROR)
            raise ValueError("Either 'id' or 'email' must be provided to update a user.")

        if not user:
            Manager.log("No user provided for update.", level=ERROR)
            raise ValueError("No user provided for update.")
        
        if not isinstance(user, UserJson):
            Manager.log("Provided user is not an instance of User.", level=ERROR)
            raise ValueError("Provided user is not an instance of User.")
        
        params = []
        
        userparam = ""
        for key, value in user.items():
            if value is not None:
                userparam += f"{key} = ?, "
                params.append(value)

        if email:
            queryparam = "email = ?"
            params = [email]

        if id:
            queryparam = "ID = ?"
            params = [id]
        
        if not userparam:
            Manager.log("No fields to update in user.", level=ERROR)
            raise ValueError("No fields to update in user.")
        
        userparam = userparam.rstrip(", ")
        
        query = f"UPDATE Users SET {userparam} WHERE {queryparam};"

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise ValueError("Failed to get cursor for user update.")

                params = tuple(params)

                cursor.execute(query, params)

                if cursor.rowcount == 0:
                    Manager.log("No user found to update.", level=ERROR)
                    raise ValueError("No user found to update.")
                
        except SQLError as err:
            Manager.log(f"Error updating user: {err}", level=ERROR)
            raise ValueError(f"Error updating user: {err}")

    @staticmethod
    def delete(**kwargs) -> None:
        id: Optional[str] = kwargs.get("id", None)
        email: Optional[EmailStr] = kwargs.get("email", None)

        if not id and not email:
            Manager.log("Either 'id' or 'email' must be provided to delete a user.", level=ERROR)
            raise ValueError("Either 'id' or 'email' must be provided to delete a user.")

        if email:
            queryparam = "email = ?"
            params = (email,)

        if id:
            queryparam = "ID = ?"
            params = (id,)

        query = f"DELETE FROM Users WHERE {queryparam};"

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise ValueError("Failed to get cursor for user deletion.")

                cursor.execute(query, params)

                if cursor.rowcount == 0:
                    Manager.log("No user found to delete.", level=ERROR)
                    raise ValueError("No user found to delete.")
                
        except SQLError as err:
            Manager.log(f"Error deleting user: {err}", level=ERROR)
            raise ValueError(f"Error deleting user: {err}")

class StatusService(Service[StationStatus]):
    """
    Service class for managing StationStatus entities in the database.
    """

    @staticmethod
    def get(**kwargs) -> Optional[StationStatus]:
        """
        Get a station status from the database.
        Args:
            - MAC (MacAddress): The MAC address of the station to retrieve.

        Returns:
            - Optional[StationStatus]: The station status object if found, None otherwise.

        Raises:
            - ValueError: If MAC is not provided | if there is an error retrieving the station status.
        """
        result: Optional[StationStatus] = None
        mac: Optional[MacAddress] = kwargs.get("MAC", None)

        if not mac:
            Manager.log(f"MAC must be provided to retrieve a station status.", level=ERROR)
            raise ValueError(f"MAC must be provided to retrieve a station status.")    

        query = "SELECT * FROM Status WHERE MAC = ?;"

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise ValueError("Failed to get cursor for station status retrieval.")

                cursor.execute(query, (mac,))
                data = cursor.fetchone()
                if data:
                    result = StationStatus(
                        MAC=data[0],
                        timestamp=data[1],
                        SHT=data[2],
                        BMP=data[3],
                        CAM=data[4],
                        WIFI=data[5]
                    )

        except SQLError as err:
            Manager.log(f"Error retrieving station status :: {err}", level=ERROR)
            raise ValueError(f"Error retrieving station status :: {err}")

        return result

    @staticmethod
    def insert(**kwargs) -> None:
        """
        Insert a station status into the database.
        Args:
            status (StationStatus): The station status to insert.
        Raises:
            ValueError: If no station status is provided | if the provided station status is not an
        """
        status: Optional[StationStatus] = kwargs.get("status", None)
        if not status:
            Manager.log("No station status provided for insertion.", level=ERROR)
            raise ValueError("No station status provided for insertion.")
        
        if not isinstance(status, StationStatus):
            Manager.log("Provided station status is not an instance of StationStatus.", level=ERROR)
            raise ValueError("Provided station status is not an instance of StationStatus.")
        
        query = "INSERT INTO Status VALUES (?, ?, ?, ?, ?, ?);"
        Manager.log(f"Inserting station status :: {status}", level=DEBUG)

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise ValueError("Failed to get cursor for station status insertion.")

                cursor.execute(query, (
                    status.MAC,
                    status.timestamp,
                    int(status.SHT),
                    int(status.BMP),
                    int(status.CAM),
                    int(status.WIFI),
                ))
            
        except SQLError as err:
            Manager.log(f"Error inserting station status: {err}", level=ERROR)
            raise ValueError(f"Error inserting station status: {err}")


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
            - ValueError: If neither mac nor id is provided | if there is an error retrieving the station.
        """
        result: Optional[Station] = None
        mac: Optional[MacAddress] = kwargs.get("MAC", None)

        if not mac:
            Manager.log(f"MAC must be provided to retrieve a station.", level=ERROR)
            raise ValueError(f"Either MAC or must be provided to retrieve a station.")    

        query = f"SELECT * FROM Stations WHERE MAC = ? LIMIT 1;"

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise ValueError("Failed to get cursor for station retrieval.")

                cursor.execute(query, (mac,))
                data = cursor.fetchone()
                if data:
                    
                    status = StatusService.get(MAC=mac)

                    result = Station(
                        MAC=mac,
                        name=data[1],
                        device_model=DeviceType.match(data[2]),
                        camera_model=CameraModel.match(data[3]),
                        firmware_version=data[4],
                        altitude=data[5],
                        latitude=Latitude(data[6]),
                        longitude=Longitude(data[7]),
                        sensors=status
                    )

        except SQLError as err:
            Manager.log(f"Error retrieving station :: {err}", level=ERROR)
            raise ValueError(f"Error retrieving station :: {err}")
        
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
        """
        results: list[Station] = []
        limit: int = kwargs.get("limit", 20)
        page: int = kwargs.get("page", 0)
        offset = page * limit
        query = "SELECT * FROM Stations ORDER BY name LIMIT ? OFFSET ?;"

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise ValueError("Failed to get cursor for station listing.")

                cursor.execute(query, (limit, offset))
                data = cursor.fetchall()
                for row in data:
                    mac = row[0]
                    status = StatusService.get(MAC=mac)
                    results.append(Station(
                        MAC=mac,
                        name=row[1],
                        device_model=DeviceType.match(row[2]),
                        camera_model=CameraModel.match(row[3]),
                        firmware_version=row[4],
                        altitude=row[5],
                        latitude=Latitude(row[6]),
                        longitude=Longitude(row[7]),
                        sensors=status
                    ))

        except SQLError as err:
            Manager.log(f"Error listing stations: {err}", level=ERROR)
            raise ValueError(f"Error listing stations: {err}")

        return results

    @staticmethod
    def insert(**kwargs) -> None:
        station: Optional[Station] = kwargs.get("station", None)
        if not station:
            Manager.log("No station provided for insertion.", level=ERROR)
            raise ValueError("No station provided for insertion.")
        
        if not isinstance(station, Station):
            Manager.log("Provided station is not an instance of Station.", level=ERROR)
            raise ValueError("Provided station is not an instance of Station.")
        
        query = "INSERT INTO Stations VALUES (?, ?, ?, ?, ?, ?, ?, ?);"
        Manager.log(f"Inserting station :: {station}", level=DEBUG)

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise ValueError("Failed to get cursor for station insertion.")

                cursor.execute(query, (
                    station.MAC,
                    station.name,
                    station.device_model.value,
                    station.camera_model.value,
                    station.firmware_version,
                    station.altitude,
                    float(station.latitude),
                    float(station.longitude),
                ))

            if station.sensors:
                StatusService.insert(status=station.sensors)
            
        except SQLError as err:
            Manager.log(f"Error inserting station: {err}", level=ERROR)
            raise ValueError(f"Error inserting station: {err}")






