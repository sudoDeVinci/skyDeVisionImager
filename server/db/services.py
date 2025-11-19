from .entities import (
    dt2str,
    str2dt,
    User,
    UserJSON,
    Station,
    StationJSON,
    StationStatus,
    Reading,
    DeviceType,
    CameraModel,
    UserRole,
    ArbitraryStringMapping,
)

from .DBManager import Manager
from pydantic import BaseModel, EmailStr, SecretStr
from pydantic_extra_types.mac_address import MacAddress
from pydantic_extra_types.coordinate import Latitude, Longitude
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, override
from logging import ERROR, DEBUG
from datetime import datetime, UTC


T = TypeVar("T", bound=BaseModel)


def logcursorfailure() -> None:
    Manager.log("Failed to get cursor.", level=ERROR)


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
    def get(**kwargs: ArbitraryStringMapping) -> T | None:
        """
        Get all entities from the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def list(**kwargs: ArbitraryStringMapping) -> List[T]:
        """
        Get a slice of Entities from the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def insert(**kwargs: ArbitraryStringMapping) -> None:
        """
        Add an entity to the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def update(**kwargs: ArbitraryStringMapping) -> None:
        """
        Update an entity in the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def delete(**kwargs: ArbitraryStringMapping) -> None:
        """
        Delete an entity from the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def exists(**kwargs: ArbitraryStringMapping) -> bool:
        """
        Check if an entity exists in the database.
        """
        pass


class UserService(Service[User]):
    @override
    @staticmethod
    def get(**kwargs: ArbitraryStringMapping) -> User | None:
        """
        Get a user from the database.
        Args:
            - email (str): The email of the user to retrieve.
            - id (str): The ID of the user to retrieve.

        Returns:
            - Optional[User]: The user object if found, None otherwise.

        Raises:
            InvalidInputError: If neither email nor id is provided.
            NotFoundError: If the user is not found.
            InternalDBError: If there is an error getting the cursor or other random error.
        """

        result: User | None = None
        email: EmailStr | None = kwargs.get("email", None)
        id: str | None = kwargs.get("id", None)

        if not email and not id:
            Manager.log(
                "Either 'email' or 'id' must be provided to retrieve a user.", ERROR
            )
            raise InvalidInputError(
                "Either 'email' or 'id' must be provided to retrieve a user."
            )

        queryparam = "email = ?;" if email else "ID = ?;"
        params = (email,) if email else (id,)

        query = f"SELECT * FROM users WHERE {queryparam}"

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure("Failed to get cursor for user retrieval.")
                    raise InternalDBError("Failed to get cursor for user retrieval.")

                _ = cursor.execute(query, params)
                data: ArbitraryStringMapping = cursor.fetchone()
                if not data:
                    raise NotFoundError("User not found.")
                if data:
                    result = User(**data)

        except Exception as e:
            Manager.log(f"Failed to retrieve user: {e}", ERROR)
            raise InternalDBError(f"Failed to retrieve user: {e}")

        return result

    @override
    @staticmethod
    def list(**kwargs: ArbitraryStringMapping) -> list[User]:
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
        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    raise InternalDBError("Failed to get cursor for user listing.")

                _ = cursor.execute(query, (limit, offset))
                data = cursor.fetchall()
                results = (
                    User(
                        ID=row[0],
                        name=row[1],
                        email=row[2],
                        password=row[3],
                        role=UserRole.match(row[4]),
                    )
                    for row in data
                )
        except Exception as e:
            Manager.log(f"Failed to fetch users: {e}", ERROR)
            raise InternalDBError("Failed to fetch users.")

        return results

    @override
    @staticmethod
    def insert(**kwargs: ArbitraryStringMapping) -> None:
        """
        Insert a user into the database.
        Args:
            user (User): The user to insert.
        Raises:
            InvalidInputError: If no user is provided | if the provided user is not an instance of User.
            InternalDBError: If there is an error getting the cursor for user insertion.
        """
        user: User | None = kwargs.get("user", None)
        if not user:
            Manager.log("No user provided for insertion.", level=ERROR)
            raise InvalidInputError("No user provided for insertion.")

        if not isinstance(user, User):
            Manager.log("Provided user is not an instance of User.", level=ERROR)
            raise InvalidInputError("Provided user is not an instance of User.")

        query = "INSERT INTO Users VALUES (?, ?, ?, ?, ?);"
        Manager.log(f"Inserting user :: {user}", level=DEBUG)

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    raise InternalDBError("Failed to get cursor for user insertion.")

                _ = cursor.execute(
                    query,
                    (
                        user.ID,
                        user.name,
                        user.email,
                        Manager.hash_password(
                            password=user.password.get_secret_value()
                        ),
                        user.role.value,
                    ),
                )

        except Exception as e:
            Manager.log(f"Failed to insert user :: {user} :: {e}", level=ERROR)
            raise InternalDBError(f"Failed to insert user :: {user} :: {e}")

    @staticmethod
    def insert_batch(**kwargs: ArbitraryStringMapping) -> None:
        """
        Inserts a batch of users into the database.

        Args:
            users (list[User]): A list of User objects to be inserted.

        Raises:
            InvalidInputError: If no users are provided for batch insertion.
            InternalDBError: If the database operation fails.
        """

        users: list[User] = kwargs.get("users", [])
        query: str = "INSERT INTO Users VALUES (?, ?, ?, ?, ?);"
        if not users:
            raise InvalidInputError("No users provided for batch insertion.")

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    raise InternalDBError(
                        "Failed to get cursor for user batch insertion"
                    )

                params: tuple[str, str, str, str, str] = (
                    (
                        user.ID,
                        user.name,
                        user.email,
                        Manager.hash_password(
                            password=user.password.get_secret_value()
                        ),
                        user.role.value,
                    )
                    for user in users
                )
                _ = cursor.executemany(query, params)

        except Exception as e:
            raise InternalDBError(f"Failed to insert users: {e}")

    @override
    @staticmethod
    def update(**kwargs: ArbitraryStringMapping) -> None:
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
        id: str | None = kwargs.get("id", None)
        email: EmailStr | None = kwargs.get("email", None)
        user: UserJSON | None = kwargs.get("user", None)

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
    def update_batch(**kwargs: ArbitraryStringMapping) -> None:
        """
        Update multiple users in the database using batch processing.
        Assumes all users are updating the same fields.

        Args:
            users (List[tuple[str, str, UserJSON]]): A list of tuples containing the user's ID, email, and JSON data.
        Raises:
            InvalidInputError: If neither id nor email is provided | if no user data is provided
            InternalDBError: If there is an error getting the cursor for user update.
            NotFoundError: If no user is found to update.
        """

        users: List[tuple[str, str, UserJSON]] | None = kwargs.get("users", None)
        if not users:
            raise InvalidInputError("No user data provided.")

        # Get fields from the first user (assuming all have the same fields)
        first_id, first_email, first_userjson = users[0]

        if not first_userjson:
            raise InvalidInputError("No fields to update.")

        # Build the SET clause from the first user's fields
        fields = []
        for key in first_userjson.keys():
            fields.append(key)

        userparam = ", ".join(f"{field} = ?" for field in fields)

        # Determine if we're using ID or email for all users
        use_id = bool(first_id)
        queryparam = "ID = ?" if use_id else "email = ?"

        query = f"UPDATE Users SET {userparam} WHERE {queryparam};"

        # Build parameter tuples for each user
        params_list = []
        for id, email, userjson in users:
            params = []

            for field in fields:
                value = userjson.get(field)

                if isinstance(value, UserRole):
                    params.append(value.value)
                elif isinstance(value, SecretStr):
                    params.append(Manager.hash_password(value.get_secret_value()))
                else:
                    params.append(value)

            # Add identifier (ID or email) at the end
            params.append(id if use_id else email)
            params_list.append(tuple(params))

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    raise InternalDBError("Failed to get cursor for user batch update.")

                cursor.executemany(query, params_list)

                if cursor.rowcount == 0:
                    raise NotFoundError("No users found to update.")

        except Exception as e:
            Manager.log.error(f"Failed to update users: {e}", ERROR)
            raise InternalDBError(f"Failed to update users: {e}")

    @override
    @staticmethod
    def delete(**kwargs: ArbitraryStringMapping) -> None:
        """
        Delete a user from the database.
        Args:
            id (str): The ID of the user to delete.
            email (EmailStr): The email of the user to delete.
        Raises:
            InvalidInputError: If neither id nor email is provided.
            InternalDBError: If there is an error getting the cursor for user deletion.
            NotFoundError: If no user is found to delete.
        """
        id: str | None = kwargs.get("id", None)
        email: EmailStr | None = kwargs.get("email", None)

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

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    raise InternalDBError("Failed to get cursor for user deletion.")

                _ = cursor.execute(query, params)

                if cursor.rowcount == 0:
                    raise NotFoundError("No user found to delete.")
        except Exception as e:
            raise InternalDBError(f"Failed to delete user: {e}")

    @override
    @staticmethod
    def exists(**kwargs: ArbitraryStringMapping) -> bool:
        id: str | None = kwargs.get("id", None)
        email: str | None = kwargs.get("email", None)
        if id is None and email is None:
            raise InvalidInputError(
                "Either 'id' or 'email' must be provided to check if a user exists."
            )

        if id:
            queryparam = "ID = ?"
            params = (id,)
        elif email:
            queryparam = "email = ?"
            params = (email,)
        else:
            raise InvalidInputError(
                "Either 'id' or 'email' must be provided to check if a user exists."
            )

        query = f"SELECT COUNT(*) FROM Users WHERE {queryparam};"
        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    raise InternalDBError(
                        "Failed to get cursor for user existence check."
                    )
                _ = cursor.execute(query, params)
                result = cursor.fetchone()
                return result[0] > 0 if result else False
        except Exception as e:
            raise InternalDBError(f"Failed to check user existence: {e}")


class StatusService(Service[StationStatus]):
    """
    Service class for managing StationStatus entities in the database.
    """

    @override
    @staticmethod
    def get(**kwargs: ArbitraryStringMapping) -> StationStatus | None:
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
        result: StationStatus | None = None
        mac: MacAddress | None = kwargs.get("MAC", None)

        if not mac:
            Manager.log(
                "MAC must be provided to retrieve a station status.", level=ERROR
            )
            raise InvalidInputError(
                "MAC must be provided to retrieve a station status."
            )

        query = "SELECT * FROM Status WHERE MAC = ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError(
                    "Failed to get cursor for station status retrieval."
                )

            _ = cursor.execute(query, (mac,))
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

    @override
    @staticmethod
    def insert(**kwargs: ArbitraryStringMapping) -> None:
        """
        Insert a station status into the database.
        Args:
            status (StationStatus): The station status to insert.
        Raises:
            InvalidInputError: If no station status is provided | if the provided station status is not an instance of StationStatus.
            InternalDBError: If there is an error getting the cursor for station status insertion.
            SQLError: If there is an error executing the SQL query.
        """
        status: StationStatus | None = kwargs.get("status", None)
        if not status:
            raise InvalidInputError("No station status provided for insertion.")

        query = "INSERT INTO Status VALUES (?, ?, ?, ?, ?, ?);"
        Manager.log(f"Inserting station status :: {status}", level=DEBUG)

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError(
                    "Failed to get cursor for station status insertion."
                )

            _ = cursor.execute(
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
    def insert_batch(**kwargs: ArbitraryStringMapping) -> None:
        """
        Insert a station status into the database.
        Args:
            statuses (list[StationStatus]): The station statuses to insert.
        Raises:
            InvalidInputError: If no station status is provided | if the provided station status is not an instance of StationStatus.
            InternalDBError: If there is an error getting the cursor for station status insertion.
            SQLError: If there is an error executing the SQL query.
        """
        statuses: list[StationStatus] = kwargs.get("statuses", None)
        if not statuses:
            raise InvalidInputError("No station statuses provided for insertion.")

        query = "INSERT INTO Status VALUES (?, ?, ?, ?, ?, ?);"
        Manager.log(f"Inserting station statuses :: {statuses}", level=DEBUG)

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError(
                    "Failed to get cursor for station status batch insertion."
                )

            params = (
                (
                    status.MAC,
                    dt2str(status.timestamp),
                    int(status.SHT),
                    int(status.BMP),
                    int(status.CAM),
                    int(status.WIFI),
                )
                for status in statuses
            )

            _ = cursor.executemany(query, params)

    @override
    @staticmethod
    def list(**kwargs: ArbitraryStringMapping) -> list[StationStatus]:
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

            _ = cursor.execute(query, (limit, offset))
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

    @override
    @staticmethod
    def update(**kwargs: ArbitraryStringMapping) -> None:
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
        mac: MacAddress | None = kwargs.get("MAC", None)
        status: StationStatus | None = kwargs.get("status", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to update a station status.")

        if not status:
            raise InvalidInputError("No station status provided for update.")

        query = "UPDATE Status SET timestamp = ?, SHT = ?, BMP = ?, CAM = ?, WIFI = ? WHERE MAC = ?;"
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

            _ = cursor.execute(query, params)

            if cursor.rowcount == 0:
                raise NotFoundError("No station status found to update.")

    @staticmethod
    def update_batch(**kwargs: ArbitraryStringMapping) -> None:
        """
        Update multiple station statuses in the database using batch processing.

        Args:
            statuses (List[tuple[MacAddress, StationStatus]]): A list of tuples containing the MAC address and StationStatus data.
        Raises:
            InvalidInputError: If no station status data is provided.
            InternalDBError: If there is an error getting the cursor for station status update.
            SQLError: If there is an error executing the SQL query.
            NotFoundError: If no station status is found to update.
        """
        statuses: List[tuple[MacAddress, StationStatus]] | None = kwargs.get(
            "statuses", None
        )
        if not statuses:
            raise InvalidInputError("No station statuses provided for update.")

        query = "UPDATE Status SET timestamp = ?, SHT = ?, BMP = ?, CAM = ?, WIFI = ? WHERE MAC = ?;"
        params = [
            (
                status.timestamp,
                status.SHT,
                status.BMP,
                status.CAM,
                status.WIFI,
                mac,
            )
            for mac, status in statuses
        ]

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError(
                    "Failed to get cursor for station status batch update."
                )

            _ = cursor.executemany(query, params)

            if cursor.rowcount == 0:
                raise NotFoundError("No station statuses found to update.")

    @override
    @staticmethod
    def delete(**kwargs: ArbitraryStringMapping) -> None:
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
        mac: MacAddress | None = kwargs.get("MAC", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to delete a station status.")

        query = "DELETE FROM Status WHERE MAC = ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                logcursorfailure()
                raise InternalDBError(
                    "Failed to get cursor for station status deletion."
                )

            _ = cursor.execute(query, (mac,))

            if cursor.rowcount == 0:
                Manager.log("No station status found to delete.", level=ERROR)
                raise NotFoundError("No station status found to delete.")

    @staticmethod
    def delete_batch() -> None:
        """
        Delete a list of station statuses from the database.
        Args:
            MACS (ListMacAddress]): The MAC addresses of the station statuses to delete.
        Raises:
            InvalidInputError: If MAC is not provided.
            InternalDBError: If there is an error getting the cursor for station status deletion.
            SQLError: If there is an error executing the SQL query.
            NotFoundError: If no station status is found to delete.
        """

        macs: List[MacAddress] = kwargs.get("MACS", [])

        if not macs:
            raise InvalidInputError("MAC must be provided to delete a station status.")

        query = "DELETE FROM Status WHERE MAC = ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                logcursorfailure()
                raise InternalDBError(
                    "Failed to get cursor for station status batch-deletion."
                )

            _ = cursor.executemany(query, macs)

            if cursor.rowcount == 0:
                Manager.log("No station status found to delete.", level=ERROR)
                raise NotFoundError("No station status found to delete.")

    @override
    @staticmethod
    def exists(**kwargs: ArbitraryStringMapping) -> bool:
        """
        Check if a station status exists in the database.
        Args:
            MAC (MacAddress): The MAC address of the station status to check.
        Returns:
            bool: True if the station status exists, False otherwise.
        Raises:
            InvalidInputError: If MAC is not provided.
            InternalDBError: If there is an error getting the cursor for existence check.
            SQLError: If there is an error executing the SQL query.
        """
        mac: MacAddress | None = kwargs.get("MAC", None)

        if not mac:
            raise InvalidInputError(
                "MAC must be provided to check station status existence."
            )

        query = "SELECT 1 FROM Status WHERE MAC = ? LIMIT 1;"

        with Manager.cursor() as cursor:
            if not cursor:
                logcursorfailure()
                raise InternalDBError("Failed to get cursor for existence check.")

            _ = cursor.execute(query, (mac,))
            return cursor.fetchone() is not None

    @staticmethod
    def exists_batch(**kwargs) -> None:
        """
        Check if a set of station statuses exist in the database.
        Args:
            MACS (MacAddress): The MAC address of the station status to check.
        Returns:
            bool: True if the station status exists, False otherwise.
        Raises:
            InvalidInputError: If MAC is not provided.
            InternalDBError: If there is an error getting the cursor for existence check.
            SQLError: If there is an error executing the SQL query.
        """
        macs: List[StationStatus] = kwargs.get("MACS", [])

        if not macs:
            raise InvalidInputError(
                "MAC must be provided to check station status existence."
            )

        length = len(macs)
        placeholders: str = ",".join(["?"] * length)
        query = f"SELECT MAC FROM Status WHERE MAC IN ({placeholders});"

        with Manager.cursor as cursor:
            if not cursor:
                logcursorfailure()
                raise InternalDBError("Failed to get cursor for existence check.")

            cursor.execute(query, macs)
            found = set(cursor.fetchall())
            results: List[bool] = [(item in found) for item in macs]
            return results


class StationService(Service[Station]):
    """
    Service class for managing Station entities in the database.
    """

    @override
    @staticmethod
    def get(**kwargs: ArbitraryStringMapping) -> Station:
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
        result: Station | None = None
        mac: MacAddress | None = kwargs.get("MAC", None)

        if not mac:
            raise InvalidInputError(
                "Either MAC or must be provided to retrieve a station."
            )

        query = "SELECT * FROM Stations WHERE MAC = ? LIMIT 1;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for station retrieval.")

            _ = cursor.execute(query, (mac,))
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

    @override
    @staticmethod
    def list(**kwargs: ArbitraryStringMapping) -> list[Station]:
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

            _ = cursor.execute(query, (limit, offset))
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

    @override
    @staticmethod
    def insert(**kwargs: ArbitraryStringMapping) -> None:
        """
        Insert a station into the database.
        Args:
            station (Station): The station to insert.
        Raises:
            InvalidInputError: If no station is provided | if the provided station is not an instance of Station.
            InternalDBError: If there is an error getting the cursor for station insertion.
            SQLError: If there is an error executing the SQL query.
        """
        station: Station | None = kwargs.get("station", None)
        if not station:
            raise InvalidInputError("No station provided for insertion.")

        query = "INSERT INTO Stations VALUES (?, ?, ?, ?, ?, ?, ?, ?);"
        Manager.log(f"Inserting station :: {station}", level=DEBUG)

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for station insertion.")

            _ = cursor.execute(
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

        if station.sensors is None:
            station.sensors = StationStatus(
                MAC=station.MAC,
                timestamp=datetime.now(tz=UTC),
                SHT=False,
                BMP=False,
                CAM=False,
                WIFI=False,
            )
        StatusService.insert(status=station.sensors)

    @override
    @staticmethod
    def update(**kwargs: ArbitraryStringMapping) -> None:
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
        mac: MacAddress | None = kwargs.get("MAC", None)
        station: StationJSON | None = kwargs.get("station", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to update a station.")

        if not station:
            raise InvalidInputError("No station provided for update.")

        _ = station.pop("sensors", None)
        params: List[object] = []
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
            _ = cursor.execute(query, paramstup)

            if cursor.rowcount == 0:
                raise NotFoundError("No station found to update.")

    @override
    @staticmethod
    def delete(**kwargs: ArbitraryStringMapping) -> None:
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
        mac: MacAddress | None = kwargs.get("MAC", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to delete a station.")

        query = "DELETE FROM Stations WHERE MAC = ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for station deletion.")

            StatusService.delete(MAC=mac)
            _ = cursor.execute(query, (mac,))

            if cursor.rowcount == 0:
                raise NotFoundError("No station found to delete.")

    @override
    @staticmethod
    def exists(**kwargs: ArbitraryStringMapping) -> bool:
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
        mac: MacAddress | None = kwargs.get("MAC", None)

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

            _ = cursor.execute(query, (mac,))
            return cursor.fetchone() is not None


class ReadingService(Service[Reading]):
    """
    Service class for managing Reading entities in the database.
    """

    @staticmethod
    def get(**kwargs: ArbitraryStringMapping) -> Reading | None:
        """
        Get a reading from the database.
        Args:
            - MAC (MacAddress): The MAC address of the station the reading belongs to.
            - timestamp (datetime): The timestamp of the reading to retrieve.

        Returns:
            Optional[Reading]: The reading object if found, None otherwise.

        Raises:
            InvalidInputError: If id is not provided | if there is an error retrieving the reading.
            InternalDBError: If there is an error getting the cursor for reading retrieval.
            SQLError: If there is an error executing the SQL query.
        """
        result: Reading | None = None
        mac: MacAddress | None = kwargs.get("MAC", None)
        timestamp: datetime | None = kwargs.get("timestamp", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to retrieve a reading.")
        if not timestamp:
            raise InvalidInputError("timestamp must be provided to retrieve a reading.")

        query = "SELECT * FROM Readings WHERE MAC = ? AND timestamp = ? LIMIT 1;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for reading retrieval.")

            _ = cursor.execute(query, (mac, dt2str(timestamp)))
            data = cursor.fetchone()
            if data:
                result = Reading(
                    MAC=data[0],
                    timestamp=str2dt(data[1]),
                    temperature=data[2],
                    humidity=data[3],
                    pressure=data[4],
                    dewpoint=data[5],
                    filepath=data[6],
                )

        return result

    @staticmethod
    def list(**kwargs: ArbitraryStringMapping) -> list[Reading]:
        """
        Get a slice of readings from the database for a given station.
        Args:
            limit (int): The maximum number of readings to retrieve.
            page (int): The page number to retrieve (0-indexed).
            MAC (MacAddress): The MAC address of the station to filter readings by.
            start (datetime): The start timestamp to filter readings by.
            end (datetime): The end timestamp to filter readings by.

        Returns:
            list[Reading]: A list of reading objects.
        Raises:
            InternalDBError: If there is an error getting the cursor for reading listing.
            SQLError: If there is an error executing the SQL query.
        """
        results: list[Reading] = []
        limit: int = kwargs.get("limit", 20)
        page: int = kwargs.get("page", 0)
        mac: MacAddress | None = kwargs.get("MAC", None)
        start: datetime | None = kwargs.get("start", None)
        end: datetime | None = kwargs.get("end", None)
        offset = page * limit

        if not mac:
            raise InvalidInputError("MAC must be provided to list readings.")

        queryparam = "MAC = ?"
        params: list[MacAddress] = [mac]

        if start:
            queryparam += " AND timestamp >= ?"
            params.append(dt2str(start))

        if end:
            queryparam += " AND timestamp <= ?"
            params.append(dt2str(end))

        query = f"SELECT * FROM Readings WHERE {queryparam} ORDER BY timestamp DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for reading listing.")

            _ = cursor.execute(query, tuple(params))
            data = cursor.fetchall()
            for row in data:
                results.append(
                    Reading(
                        MAC=row[0],
                        timestamp=str2dt(row[1]),
                        temperature=row[2],
                        humidity=row[3],
                        pressure=row[4],
                        dewpoint=row[5],
                        filepath=row[6],
                    )
                )

        return results

    @staticmethod
    def insert(**kwargs: ArbitraryStringMapping) -> None:
        """
        Insert a reading into the database.
        Args:
            reading (Reading): The reading to insert.
        Raises:
            InvalidInputError: If no reading is provided | if the provided reading is not an instance of Reading.
            InternalDBError: If there is an error getting the cursor for reading insertion.
            SQLError: If there is an error executing the SQL query.
        """
        reading: Reading | None = kwargs.get("reading", None)
        if not reading:
            raise InvalidInputError("No reading provided for insertion.")

        if not isinstance(reading, Reading):
            raise InvalidInputError("Provided reading is not an instance of Reading.")

        query = "INSERT INTO Readings VALUES (?, ?, ?, ?, ?, ?, ?);"
        Manager.log(f"Inserting reading :: {reading}", level=DEBUG)

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for reading insertion.")

            _ = cursor.execute(
                query,
                (
                    reading.MAC,
                    dt2str(reading.timestamp),
                    reading.temperature,
                    reading.humidity,
                    reading.pressure,
                    reading.dewpoint,
                    reading.filepath,
                ),
            )

    @staticmethod
    def update(**kwargs: ArbitraryStringMapping) -> None:
        """
        Update a reading in the database.
        Args:
            MAC (MacAddress): The MAC address of the station the reading belongs to.
            timestamp (datetime): The timestamp of the reading to update.
            reading (ReadingJSON): The reading data to update.
        Raises:
            InvalidInputError: If MAC is not provided | if timestamp is not provided | if no reading data is provided | if the provided reading data is not an instance of ReadingJSON.
            InternalDBError: If there is an error getting the cursor for reading update.
            SQLError: If there is an error executing the SQL query.
            NotFoundError: If no reading is found to update.
        """
        mac: MacAddress | None = kwargs.get("MAC", None)
        timestamp: datetime | None = kwargs.get("timestamp", None)
        reading: Reading | None = kwargs.get("reading", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to update a reading.")
        if not timestamp:
            raise InvalidInputError("timestamp must be provided to update a reading.")
        if not reading:
            raise InvalidInputError("No reading provided for update.")

        query = "UPDATE Readings SET temperature = ?, humidity = ?, pressure = ?, dewpoint = ?, filepath = ? WHERE MAC = ? AND timestamp = ?;"
        params = (
            reading.temperature,
            reading.humidity,
            reading.pressure,
            reading.dewpoint,
            reading.filepath,
            mac,
            dt2str(timestamp),
        )

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for reading update.")

            _ = cursor.execute(query, params)

            if cursor.rowcount == 0:
                raise NotFoundError("No reading found to update.")

    @staticmethod
    def delete(**kwargs) -> None:
        """
        Delete a reading from the database.
        Args:
            MAC (MacAddress): The MAC address of the station the reading belongs to.
            timestamp (datetime): The timestamp of the reading to delete.
        Raises:
            InvalidInputError: If MAC is not provided | if timestamp is not provided.
            InternalDBError: If there is an error getting the cursor for reading deletion.
            SQLError: If there is an error executing the SQL query.
            NotFoundError: If no reading is found to delete.
        """
        mac: MacAddress | None = kwargs.get("MAC", None)
        timestamp: datetime | None = kwargs.get("timestamp", None)

        if not mac:
            raise InvalidInputError("MAC must be provided to delete a reading.")

        if not timestamp:
            raise InvalidInputError("timestamp must be provided to delete a reading.")

        query = "DELETE FROM Readings WHERE MAC = ? AND timestamp = ?;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError("Failed to get cursor for reading deletion.")

            _ = cursor.execute(query, (mac, dt2str(timestamp)))

            if cursor.rowcount == 0:
                raise NotFoundError("No reading found to delete.")

    @staticmethod
    def exists(**kwargs: ArbitraryStringMapping) -> bool:
        """
        Check if a reading exists in the database.
        Args:
            MAC (MacAddress): The MAC address of the station the reading belongs to.
            timestamp (datetime): The timestamp of the reading to check.
        Returns:
            bool: True if the reading exists, False otherwise.
        Raises:
            InvalidInputError: If MAC is not provided | if timestamp is not provided.
            InternalDBError: If there is an error getting the cursor for reading existence check.
            SQLError: If there is an error executing the SQL query.
        """
        mac: MacAddress | None = kwargs.get("MAC", None)
        timestamp: datetime | None = kwargs.get("timestamp", None)

        if not mac:
            raise InvalidInputError(
                "MAC must be provided to check if a reading exists."
            )
        if not timestamp:
            raise InvalidInputError(
                "timestamp must be provided to check if a reading exists."
            )

        query = "SELECT 1 FROM Readings WHERE MAC = ? AND timestamp = ? LIMIT 1;"

        with Manager.cursor() as cursor:
            if not cursor:
                raise InternalDBError(
                    "Failed to get cursor for reading existence check."
                )

            _ = cursor.execute(query, (mac, dt2str(timestamp)))
            return cursor.fetchone() is not None
