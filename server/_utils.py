from werkzeug.datastructures import Headers
from flask import Request
from enum import Enum
from functools import lru_cache
from re import compile
from typing_extensions import Self, Iterable, Any, Optional


class MissingHeadersError(Exception):
    """
    Exception raised when a required header is missing.
    """

    def __init__(self: "MissingHeadersError", headers: Iterable[str]):
        super().__init__(f"Missing required headers: {headers}")


class HEADERS(Enum):
    """
    Expected Headers to the /api route from esp devices.
    """

    MACADDRESS = "X-MAC-Address"
    TIMESTAMP = "X-Timestamp"
    FIRMWAREVERSION = "X-Firmware-Version"
    CONTENTTYPE = "Content-Type"
    CONTENTLENGTH = "Content-Length"
    USERAGENT = "User-Agent"
    UNKNOWN = "UNKNOWN"

    @classmethod
    @lru_cache(maxsize=10)
    def match(cls: "HEADERS", header: Optional[str]) -> Self:  # type: ignore
        """
        Match input string to header.
        """
        return cls[header] if header in cls.__members__.items() else cls.UNKNOWN  # type: ignore

    @classmethod
    @lru_cache(maxsize=10)
    def __contains__(cls: "HEADERS", header: Optional[str]) -> bool:  # type: ignore
        """
        Check if a header is supported.
        """
        return HEADERS.match(header) != cls.UNKNOWN

    @classmethod
    @lru_cache(maxsize=10)
    def _missing_(cls: "HEADERS", value: Any):  # type: ignore
        """
        Handle missing headers.
        """
        return cls.UNKNOWN


def headercheck(headers: Headers, required: Optional[Iterable[str]] = None) -> None:
    """
    Check if all required headers are present in the request.

    Args:
        required (Iterable[str]): List of required header names.
        headers (Headers): Headers from the request.
    Raises:
        MissingHeadersError: If any required headers are missing.
    """
    if required is None:
        required = [
            HEADERS.MACADDRESS.value,
            HEADERS.TIMESTAMP.value,
            HEADERS.FIRMWAREVERSION.value,
        ]
    missing = [head for head in required if head not in headers]
    if missing:
        raise MissingHeadersError(missing)
