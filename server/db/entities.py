from typing import (
    TypeVar,
    TypedDict,
    Optional,
    Final,
    Any,
    Generic,
    List,
    Mapping,
    Sequence,
)
from datetime import datetime
from decimal import Decimal
from abc import ABC, abstractmethod
from pydantic import BaseModel, EmailStr
from pydantic_extra_types.mac_address import MacAddress
from pydantic_extra_types.coordinate import Latitude, Longitude
from enum import Enum
from functools import lru_cache


def dt2str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


def str2dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")


AnyDict = TypeVar(
    "AnyDict",
    bound=dict[
        str,
        bytes
        | bytearray
        | str
        | int
        | Decimal
        | bool
        | set[int]
        | set[Decimal]
        | set[str]
        | set[bytes]
        | set[bytearray]
        | Sequence[Any]
        | Mapping[str, Any]
        | None,
    ],
)


class ErrorDict(TypedDict, total=False):
    """
    Schema for error responses.
    The exct meaning for each field are loose
    and are not strictly defined.
    This is mostly intended to mirror the structure
    of the error responses returned by the Ebanq API.
    """

    title: str
    details: str
    code: str
    source: str
    target: str
    meta: dict[str, Any]


class ErrorResponse(TypedDict, total=False):
    """
    Errors and warning response schema.
    This mirrors the structure of the error responses
    returned by the Ebanq API.
    It contains a list of errors and warnings encountered during the request.
    The `timestamp` field is used to indicate when the response was generated.
    """

    errors: List[ErrorDict]
    warnings: List[ErrorDict]
    timestamp: str


class EbanqResponse(TypedDict, Generic[AnyDict], total=False):
    data: AnyDict
    timestamp: str


class CameraModel(Enum):
    UNKNOWN = "UNKNOWN"
    OV5640 = "OV5640"


class DeviceType(Enum):
    UNKNOWN = "UNKNOWN"
    ESP32S3 = "ESP32S3"


class EntityJSON(TypedDict, total=False):
    uid: str
    created: str
    edited: str


class UserRole(Enum):
    ADMIN = "admin"
    MEMBER = "member"
    VISITOR = "visitor"
    UNKNOWN = "unknown"

    @classmethod
    @lru_cache(maxsize=None)
    def match(cls, role: str):
        """
        Match input string to user role.
        """
        role = role.lower()
        return UserRole[role] if role in cls.__members__.items() else cls.UNKNOWN

    @classmethod
    @lru_cache(maxsize=None)
    def __contains__(cls, role: str) -> bool:
        """
        Check if a role is present in the enum.
        """
        return role.lower() in cls.__members__.values()


class Entity(BaseModel, ABC):
    """
    Abstract Parent class representing a given row in a db table, either devices, sensors or readings.

    Attributes:

        MAC (str): The MAC address of the device.
        timestamp (str): The timestamp of the object data.
    """

    MAC: MacAddress
    timestamp: datetime


class StationStatus(Entity):
    """
    A representation of the state of the sensor onboard a given weather station
    at a given timestamp. Equivalent to a single row in the sensor table.

    Attributes:
        sht (bool): Status of the SHT31-D.
        bmp (bool): Status of the BMP280.
        cam (bool): Status of the camera.
        wifi (bool): Status of the WiFi connection.
    """

    SHT: bool
    BMP: bool
    CAM: bool
    WIFI: bool


class Station(Entity):
    """
    Represents a station in the database.
    """

    name: str
    device_model: str
    device_type: DeviceType
    camera_model: CameraModel
    firmware_version: str
    altitude: float
    latitude: Latitude
    longitude: Longitude
    sensors: Optional[list[StationStatus]] = None


class Reading(Entity):
    """
    Represents a reading from a device in the database.
    """

    temperature: float
    humidity: float
    pressure: float
    dewpoint: float
    filepath: Optional[str] = None


class Location(BaseModel):
    """
    Represents a location in the database.
    """

    country: str
    region: str
    city: str
    latitude: Latitude
    longitude: Longitude


class User(BaseModel):
    """
    Represents a user in the database.
    """

    ID: str
    name: str
    email: EmailStr
    role: UserRole = UserRole.VISITOR
