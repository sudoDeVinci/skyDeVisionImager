from .entities import (
    User,
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
from bcrypt import hashpw, gensalt, checkpw
from logging import getLogger, ERROR

T = TypeVar("T", bound=BaseModel)


logcursorfailure = lambda: Manager.log("Failed to get cursor.", level=ERROR)


def hash_password(password: str, secret: str) -> str:
    """
    Hash + salt a password via bcrypt using the database secret key.

    Args:
        - password(str): The password to hash

    Returns:
        - str: The hashed password
    """

    hashed: bytes = b""

    try:
        peppered = f"{password}{secret}"
        hashed = hashpw(peppered.encode("utf-8"), gensalt())
    except Exception as err:
        getLogger(__name__).exception(f"Error hashing password::: {err}")
        hashed = hashpw(password.encode("utf-8"), gensalt())

    return hashed.decode("utf-8")


def password_match(stored_hash: str, provided_password: str, secret: str) -> bool:
    """
    Check if a password matches the stored hash.

    Args:
        - stored_hash(str): The stored password hash
        - provided_password(str): The password to check

    Returns:
        - bool: True if the password matches, False otherwise
    """
    try:
        peppered = f"{provided_password}{secret}"
        return checkpw(peppered.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception as err:
        getLogger(__name__).exception(f"Error checking password::: {err}")
        return checkpw(provided_password.encode("utf-8"), stored_hash.encode("utf-8"))


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
    def insert(**kwargs) -> Optional[T]:
        """
        Add an entity to the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def update(**kwargs) -> bool:
        """
        Update an entity in the database.
        """
        pass

    @staticmethod
    @abstractmethod
    def delete(**kwargs) -> bool:
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
    """
    Service class for user-related operations.
    """

    @staticmethod
    def get(**kwargs) -> Optional[User]:
        """
        Get a user from the database.
        """

        result: Optional[User] = None
        email: Optional[EmailStr] = kwargs.get("email", None)
        ID: Optional[str] = kwargs.get("ID", None)

        errors: list[str] = []

        if not email:
            if not ID:
                errors.append("email or ID is required to get a user.")

        if errors:
            Manager.log(f"Error retrieving user: " f"400 | {', '.join(errors)}")
            return None

        querybase = f"SELECT * FROM users WHERE "
        if ID:
            query = f"{querybase}ID = ?"
            params = (ID,)

        elif email:
            query = f"{querybase}email = ?"
            params = (email,)

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    return None

                cursor.execute(query, params)
                data = cursor.fetchone()
                if data:
                    result = User(**data)
                else:
                    Manager.log(f"User not found: {email or ID}", level=ERROR)

        except SQLError as err:
            Manager.log(f"Error retrieving user: {err}", level=ERROR)

        finally:
            if cursor:
                cursor.close()
        return result

    @staticmethod
    def list(**kwargs) -> list[User]:
        """
        Get a slice of users from the database.
        """
        result: list[User] = []
        query = "SELECT * FROM users"
        params = ()

        try:
            with Manager.cursor() as cursor:
                if not cursor:
                    logcursorfailure()
                    return result

                cursor.execute(query, params)
                data = cursor.fetchall()
                for row in data:
                    result.append(User(**row))

        except SQLError as err:
            Manager.log(f"Error listing users: {err}", level=ERROR)

        finally:
            if cursor:
                cursor.close()

        return result
