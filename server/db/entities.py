from enum import Enum
from decimal import Decimal
from functools import lru_cache
from datetime import datetime, UTC
from collections.abc import Mapping, Sequence
from pydantic_extra_types.mac_address import MacAddress
from typing import TypeVar, TypedDict, NotRequired, Self, override
from pydantic_extra_types.coordinate import Latitude, Longitude
from pydantic import BaseModel, EmailStr, SecretStr, model_validator


ArbitraryStringMapping = Mapping[str, object]


def dt2str(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")


def str2dt(s: str) -> datetime:
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt
    except ValueError:
        # fallback for legacy format without tz
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=UTC)


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
        | Sequence[object]
        | Mapping[str, object]
        | None,
    ],
)


class CameraModel(Enum):
    """
    Enum representing the camera models supported by the system.
    Each camera model is represented by a string value.
    Additional methods are provided to match camera strings, check for membership,
    and retrieve the names and members of the enum.
    """

    UNKNOWN = "UNKNOWN"
    OV5640 = "OV5640"
    DSLR = "DSLR"

    @classmethod
    @lru_cache(maxsize=50)
    def match(cls, camera: str):
        return cls[camera] if camera in cls.__members__ else cls.UNKNOWN  # Fixed

    @classmethod
    @lru_cache(maxsize=50)
    def __contains__(cls, camera: str) -> bool:
        """
        Check if a camera model is supported.
        """
        return cls.match(camera) != cls.UNKNOWN

    @override
    @classmethod
    def _missing_(cls, value: object):
        """
        Handle missing camera models.
        """
        return cls.UNKNOWN

    @classmethod
    @lru_cache(maxsize=1)
    def members(cls) -> tuple[Self, ...]:
        """
        Return an iterable of camera models.
        """
        return tuple(ctype for _, ctype in cls.__members__.items())

    @classmethod
    @lru_cache(maxsize=1)
    def names(cls) -> tuple[str, ...]:
        """
        Return a tuple of camera model names.
        """
        return tuple(names for names, _ in cls.__members__.items())


class DeviceType(Enum):
    """
    Enum representing the device types supported by the system.
    Each device type is represented by a string value.
    """

    UNKNOWN = "UNKNOWN"
    ESP32S3 = "ESP32S3"
    ESP32 = "ESP32"
    ESP8266 = "ESP8266"

    @classmethod
    @lru_cache(maxsize=50)
    def match(cls, device: str):
        return cls[device] if device in cls.__members__ else cls.UNKNOWN


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
        return UserRole[role] if role in cls.__members__ else cls.UNKNOWN

    @classmethod
    @lru_cache(maxsize=None)
    def __contains__(cls, role: str) -> bool:
        """
        Check if a role is present in the enum.
        """
        return role.lower() in cls.__members__.values()


class StationStatus(BaseModel):
    """
    A representation of the state of the sensor onboard a given weather station
    at a given timestamp. Equivalent to a single row in the sensor table.

    Attributes:
        MAC (MacAddress): The MAC address of the station.
        timestamp (datetime): The timestamp of the status.
        SHT (bool): Status of the SHT sensor.
        BMP (bool): Status of the BMP sensor.
        CAM (bool): Status of the camera.
        WIFI (bool): Status of the WiFi connection.
    """

    MAC: MacAddress
    timestamp: datetime | None = None
    SHT: bool = False
    BMP: bool = False
    CAM: bool = False
    WIFI: bool = False


class StationStatusJSON(TypedDict, total=False):
    """
    JSON representation of the StationStatus.
    This is used to serialize the StationStatus for API responses or storage.

    Attributes:
        MAC (MacAddress): The MAC address of the station.
        timestamp (datetime): The timestamp of the status in string format.
        SHT (bool): Status of the SHT sensor.
        BMP (bool): Status of the BMP sensor.
        CAM (bool): Status of the camera.
        WIFI (bool): Status of the WiFi connection.
    """

    MAC: MacAddress
    timestamp: datetime | None
    SHT: bool
    BMP: bool
    CAM: bool
    WIFI: bool


class Station(BaseModel):
    """
    Represents a station in the database.

    Attributes:
        MAC (MacAddress): The MAC address of the station.
        name (str): The name of the station.
        device_model (DeviceType): The type of device used for the station.
        camera_model (CameraModel): The model of the camera used in the station.
        firmware_version (str): The version of the firmware running on the station.
        altitude (float): The altitude of the station in meters.
        latitude (Latitude): The latitude of the station.
        longitude (Longitude): The longitude of the station.
        sensors (Optional[StationStatus]): The status of the sensors onboard the station.
    """

    MAC: MacAddress
    name: str
    device_model: DeviceType
    camera_model: CameraModel
    firmware_version: str
    altitude: float
    latitude: Latitude
    longitude: Longitude
    sensors: StationStatus | None = None

    @model_validator(mode="after")
    def check_device_model(self):
        if self.device_model is DeviceType.UNKNOWN:
            raise ValueError("device_model cannot be DeviceType.UNKNOWN")
        return self


class StationJSON(TypedDict, total=False):
    """
    JSON representation of the Station.

    This is used to serialize the Station for API responses or storage.
    It mirrors the structure of the Station model but uses TypedDict for JSON compatibility.

    Attributes:
        MAC (MacAddress): The MAC address of the station.
        name (str): The name of the station.
        device_model (DeviceType): The type of device used for the station.
        camera_model (CameraModel): The model of the camera used in the station.
        firmware_version (str): The version of the firmware running on the station.
        altitude (float): The altitude of the station in meters.
        latitude (Latitude): The latitude of the station.
        longitude (Longitude): The longitude of the station.
        sensors (Optional[StationStatusJSON]): The status of the sensors onboard the station.
    """

    MAC: MacAddress
    name: str
    device_model: DeviceType
    camera_model: CameraModel
    firmware_version: str
    altitude: float
    latitude: Latitude
    longitude: Longitude
    sensors: NotRequired[StationStatusJSON | None]


class Reading(BaseModel):
    """
    Represents a reading from a device in the database.

    Attributes:
        MAC (MacAddress): The MAC address of the station.
        timestamp (datetime): The timestamp of the reading.
        temperature (float): The temperature reading in degrees Celsius.
        humidity (float): The humidity reading in percentage.
        pressure (float): The pressure reading in hPa.
        dewpoint (float): The dew point reading in degrees Celsius.
        filepath (Optional[str]): The file path to the reading data, if applicable.
    """

    MAC: MacAddress
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    dewpoint: float
    filepath: str | None = None


class ReadingJSON(TypedDict, total=False):
    """
    JSON representation of the Reading.

    This is used to serialize the Reading for API responses or storage.
    It mirrors the structure of the Reading model but uses TypedDict for JSON compatibility.

    Attributes:
        MAC (MacAddress): The MAC address of the station.
        timestamp (datetime): The timestamp of the reading.
        temperature (float): The temperature reading in degrees Celsius.
        humidity (float): The humidity reading in percentage.
        pressure (float): The pressure reading in hPa.
        dewpoint (float): The dew point reading in degrees Celsius.
        filepath (Optional[str]): The file path to the reading data, if applicable.
    """

    MAC: MacAddress
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    dewpoint: float
    filepath: str | None


class Location(BaseModel):
    """
    Represents a location in the database.

    Attributes:
        country (str): The country of the location.
        region (str): The region of the location.
        city (str): The city of the location.
        latitude (Latitude): The latitude of the location.
        longitude (Longitude): The longitude of the location.
    """

    country: str
    region: str
    city: str
    latitude: Latitude
    longitude: Longitude


class LocationJSON(TypedDict, total=False):
    """
    JSON representation of the Location.

    This is used to serialize the Location for API responses or storage.
    It mirrors the structure of the Location model but uses TypedDict for JSON compatibility.

    Attributes:
        country (str): The country of the location.
        region (str): The region of the location.
        city (str): The city of the location.
        latitude (Latitude): The latitude of the location.
        longitude (Longitude): The longitude of the location.
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
    password: SecretStr
    role: UserRole = UserRole.VISITOR


class UserJSON(TypedDict, total=False):
    """
    Represents a user in JSON format.
    """

    ID: str
    name: str
    email: EmailStr
    password: SecretStr
    role: UserRole
