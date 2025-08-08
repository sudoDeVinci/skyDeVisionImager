from sqlite3 import OperationalError as SQLError
from .db import (
    str2dt,
    CameraModel,
    DeviceType,
    StationStatus,
    StationStatusJSON,
    StatusService,
    StationService,
    Station,
    StationJSON,
    DatabaseError,
    InternalDBError,
    InvalidInputError,
    NotFoundError,
    AlreadyExistsError,
)

from pydantic import ValidationError
from pydantic_extra_types.mac_address import MacAddress
from pydantic_extra_types.coordinate import Latitude, Longitude
from typing import cast

from flask import Flask, Blueprint, Response, request, jsonify

from datetime import datetime, UTC

from ._utils import (
    MissingHeadersError,
    headercheck,
    HEADERS as HEADERS,
)

from ._types import ErrorResponse, ErrorDict


apiRouter = Blueprint("api", __name__, url_prefix="/api")
"""
API endpoints for handling status and environmental reading data.
"""


@apiRouter.route("/status", methods=["POST", "PUT"])
def status() -> tuple[Response, int]:
    """
    Endpoint to handle status updates.
    """

    try:
        headers = request.headers
        headercheck(headers)

        # All the keys are uppercase. This is just to ensure consistency.
        tempstatusdict: StationStatusJSON = request.get_json()
        statusdict: StationStatusJSON = cast(
            StationStatusJSON,
            {k.upper(): v for k, v in tempstatusdict.items() if v is not None},
        )

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

    except MissingHeadersError as mhe:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Missing Headers",
                            details=str(mhe),
                            code="400",
                            source="/api/status",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            400,
        )

    except ValidationError as ve:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Validation Error",
                            details=str(ve),
                            code="422",
                            source="/api/status",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            422,
        )

    except InvalidInputError as iie:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Invalid Input",
                            details=str(iie),
                            code="400",
                            source="/api/status",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            400,
        )

    except NotFoundError as nfe:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Not Found",
                            details=str(nfe),
                            code="404",
                            source="/api/status",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            404,
        )

    except InternalDBError as ide:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Internal Database Error",
                            details=str(ide),
                            code="500",
                            source="/api/status",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            500,
        )

    except SQLError as e:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Database Error",
                            details=str(e),
                            code="500",
                            source="/api/status",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            500,
        )


@apiRouter.route("/register", methods=["POST", "PUT"])
def register() -> tuple[Response, int]:
    """
    Endpoint to handle station registration.
    """
    try:
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

    except MissingHeadersError as mhe:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Missing Headers",
                            details=str(mhe),
                            code="400",
                            source="/api/register",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            400,
        )

    except AlreadyExistsError as aee:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Already Exists",
                            details=str(aee),
                            code="409",
                            source="/api/register",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            409,
        )

    except ValidationError as ve:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Validation Error",
                            details=str(ve),
                            code="422",
                            source="/api/register",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            422,
        )

    except InvalidInputError as iie:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Invalid Input",
                            details=str(iie),
                            code="400",
                            source="/api/register",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            400,
        )

    except InternalDBError as ide:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Internal Database Error",
                            details=str(ide),
                            code="500",
                            source="/api/register",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            500,
        )

    except SQLError as e:
        return (
            jsonify(
                ErrorResponse(
                    errors=[
                        ErrorDict(
                            title="Database Error",
                            details=str(e),
                            code="500",
                            source="/api/register",
                        )
                    ],
                    timestamp=datetime.now(tz=UTC).isoformat(),
                )
            ),
            500,
        )


@apiRouter.route("/reading", methods=["POST", "PUT"])
def reading() -> tuple[Response, int]:
    """
    Endpoint to handle environmental reading updates.
    """
    return jsonify({"status": "success"}), 200


@apiRouter.route("/qnh", methods=["GET"])
def qnh() -> tuple[Response, int]:
    """
    Endpoint to handle QNH updates.
    """
    return jsonify({"status": "success"}), 200


@apiRouter.route("/version", methods=["GET"])
def version() -> tuple[Response, int]:
    """
    Endpoint to handle firmware version updates.
    """
    return jsonify({"status": "success"}), 200


@apiRouter.route("/check", methods=["GET"])
def index() -> tuple[Response, int]:
    """
    Endpoint to check if the API is running.
    """
    return jsonify({"status": "API is running"}), 200
