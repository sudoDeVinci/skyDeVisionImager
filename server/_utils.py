from werkzeug.datastructures import Headers
from flask import Request
from enum import Enum
from functools import lru_cache
from re import compile
from typing_extensions import (
    Self,
    Iterable,
    Any,
    Optional
)


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
    def match(cls: Self, header: Optional[str]) -> "HEADERS":
        """
        Match input string to header.
        """
        header = header.lower()
        for _, headertype in cls.__members__.items():
            if header == headertype.value:
                return headertype
        return cls.UNKNOWN

    @classmethod
    @lru_cache(maxsize=10)
    def __contains__(cls: Self, header: Optional[str]) -> bool:
        """
        Check if a header is supported.
        """
        return HEADERS.match(header) != cls.UNKNOWN

    @classmethod
    @lru_cache(maxsize=10)
    def _missing_(cls: Self, value: Any):
        """
        Handle missing headers.
        """
        return cls.UNKNOWN

    @classmethod
    @lru_cache(maxsize=1)
    def members(cls: Self) -> tuple[str]:
        """
        Return an iterable of headers.
        """
        return tuple(ctype for _, ctype in cls.__members__.items())

    @classmethod
    @lru_cache(maxsize=1)
    def names(cls: Self) -> tuple[str]:
        """
        Return a tuple of header names.
        """
        return tuple(names for names, _ in cls.__members__.items())
    

def headercheck(
    headers: Headers,
    required: Optional[Iterable[str]] = None
) -> None:
    """
    Check if all required headers are present in the request.

    Args:
        required (Iterable[str]): List of required header names.
        headers (Headers): Headers from the request.
    Raises:
        ValueError: If any required header is missing.
    """
    if required is None:
        required = [HEADERS.MACADDRESS, HEADERS.TIMESTAMP, HEADERS.FIRMWAREVERSION]
    missing = [head for head in required if head not in headers]
    if missing:
        raise ValueError(f"Missing required headers: {', '.join(missing)}") 
