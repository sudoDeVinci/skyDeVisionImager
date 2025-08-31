from sqlite3 import OperationalError as SQLError
from .db import (
    str2dt,
    CameraModel,
    DeviceType,
    StationStatus,
    StationStatusJSON,
    StatusService,
    StationService,
    Reading,
    ReadingJSON,
    ReadingService,
    Station,
    StationJSON,
    InternalDBError,
    InvalidInputError,
    NotFoundError,
    AlreadyExistsError,
)

from functools import wraps
from pydantic import ValidationError
from pydantic_extra_types.mac_address import MacAddress
from pydantic_extra_types.coordinate import Latitude, Longitude
from typing import cast

from flask import Blueprint, Response, request, jsonify

from datetime import datetime, UTC

from ._utils import (
    MissingHeadersError,
    headercheck,
    HEADERS as HEADERS,
)

from .metar._metar import MetarCacheLayer
from ._types import ErrorResponse, ErrorDict


MetarCache: MetarCacheLayer = MetarCacheLayer(airports=["ESMX"])

apiRouter = Blueprint("api", __name__, url_prefix="/api")
"""
API endpoints for handling status and environmental reading data.
"""

def _create_error_response(title: str, details: str, code: str, source: str) -> Response:
    return jsonify(ErrorResponse(
        errors=[ErrorDict(title=title, details=details, code=code, source=source)],
        timestamp=datetime.now(tz=UTC).isoformat()
    ))

def handle_api_db_errors(source_endpoint: str):
    """
    Decorator to handle common API and database errors and return appropriate HTTP responses.
    Args:
        source_endpoint (str): The endpoint where the error originated, used for error reporting.
    Returns:
        Callable: The decorated function with error handling.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MissingHeadersError as e:
                return _create_error_response("Missing Headers", str(e), "400", source_endpoint), 400
            except ValidationError as e:
                return _create_error_response("Validation Error", str(e), "422", source_endpoint), 422
            except AlreadyExistsError as e:
                return _create_error_response("Already Exists", str(e), "409", source_endpoint), 409
            except NotFoundError as e:
                return _create_error_response("Not Found", str(e), "404", source_endpoint), 404
            except InvalidInputError as e:
                return _create_error_response("Invalid Input", str(e), "400", source_endpoint), 400
            except InternalDBError as e:
                return _create_error_response("Internal Database Error", str(e), "500", source_endpoint), 500
            except SQLError as e:
                return _create_error_response("Database Error", str(e), "500", source_endpoint), 500
        return wrapper
    return decorator


@apiRouter.route("/status", methods=["POST", "PUT"])
@handle_api_db_errors("/api/status")
def status() -> tuple[Response, int]:
    """
    Endpoint to handle status updates.
    """

    headers = request.headers
    headercheck(headers)

    statusdict: StationStatusJSON = cast(StationStatusJSON, request.get_json())

    # Validation for update JSON
    mac: str = cast(str, headers.get(HEADERS.MACADDRESS.value))
    timestamp: str = cast(str, headers.get(HEADERS.TIMESTAMP.value))
    statusdict.update(
        {
            "MAC": MacAddress(mac),
            "timestamp": str2dt(timestamp),
        }
    )
    status = StationStatus(**statusdict)

    StatusService.update(MAC=mac, status=status)
    return jsonify({"status": "success"}), 200


@apiRouter.route("/register", methods=["POST", "PUT"])
@handle_api_db_errors("/api/register")
def register() -> tuple[Response, int]:
    """
    Endpoint to handle station registration.
    """
    headers = request.headers
    headercheck(headers)

    # Inital check to see if the station already exists
    mac = headers.get(HEADERS.MACADDRESS.value)
    if StationService.exists(MAC=mac):
        raise AlreadyExistsError(f"Station with MAC {mac} already exists.")

    stationdict: StationJSON = request.get_json()
    stationdict.pop("sensors", None)
    stationdict.update(
        {
            "MAC": MacAddress(mac),
            "camera_model": CameraModel.match(
                stationdict.get("camera_model", "UNKNOWN")
            ),
            "device_model": DeviceType.match(
                stationdict.get("device_model", "UNKNOWN")
            ),
        }
    )

    station = Station(**stationdict)  # type: ignore

    StationService.insert(station=station)
    return jsonify({"status": "success"}), 200


@apiRouter.route("/reading", methods=["POST", "PUT"])
@handle_api_db_errors("/api/reading")
def reading() -> tuple[Response, int]:
    """
    Endpoint to handle environmental reading updates.
    """


    headers = request.headers
    headercheck(headers)

    readingdict: ReadingJSON = cast(ReadingJSON, request.get_json())

    mac = headers.get(HEADERS.MACADDRESS.value)
    timestamp = cast(str, headers.get(HEADERS.TIMESTAMP.value))
    readingdict.update(
        {
            "MAC": MacAddress(mac),
            "timestamp": str2dt(timestamp),
        }
    )

    reading = Reading(**readingdict)
    ReadingService.update(MAC=mac, timestamp=timestamp, reading=reading)
    return jsonify({"status": "success"}), 200


@apiRouter.route("/qnh", methods=["GET"])
def qnh() -> tuple[Response, int]:
    """
    Endpoint to handle QNH updates.
    """

    qnh = MetarCache.get_qnh("ESMX")
    if qnh is None:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="QNH Not Available",
                            details="QNH data is currently not available.",
                            code="503",
                            source="/api/qnh",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            503,
        )

    return jsonify({"qnh": qnh}), 200


@apiRouter.route("/version", methods=["GET"])
def version() -> tuple[Response, int]:
    """
    Endpoint to handle firmware version updates.
    """
    return jsonify({"status": "success"}), 200


@apiRouter.route("/check", methods=["GET"])
def check() -> tuple[Response, int]:
    """
    Endpoint to check if the API is running.
    """
    return jsonify({"status": "API is running"}), 200
