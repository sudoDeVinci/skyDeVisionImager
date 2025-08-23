from typing import no_type_check
from flask.testing import FlaskClient
from server.db import (
    CameraModel,
    DeviceType,
    Station,
    StationJSON,
    StationStatus,
    StationStatusJSON,
    StationService,
    NotFoundError,
    str2dt,
    dt2str,
)
from server._utils import HEADERS
from datetime import datetime, UTC


@no_type_check
def test_post_status_successful(client: FlaskClient) -> None:

    # First we ensure the station actually exists.
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=StationStatus(
            MAC="00:1A:2B:3C:4D:5E",
            timestamp=datetime.now(tz=UTC),
            SHT=False,
            BMP=False,
            CAM=False,
            WIFI=False,
        ),
    )

    StationService.insert(station=station)

    # No error during insertion means we're good.

    response = client.post(
        "/api/status",
        json=station.sensors.model_dump(mode="json"),
        headers={
            HEADERS.TIMESTAMP.value: dt2str(datetime.now(tz=UTC)),
            HEADERS.MACADDRESS.value: station.MAC,
            HEADERS.FIRMWAREVERSION.value: "1.0.0",
        },
    )

    assert (
        response.status_code == 200
    ), f"Operation failed: {response.status_code} :: {response.get_json()}"
    data = response.get_json()
    assert data.get("status", None), "status not returned in message"


@no_type_check
def test_post_status_missing_headers(client: FlaskClient) -> None:
    response = client.post("/api/status", json={}, headers={})

    assert (
        response.status_code == 400
    ), f"Expected 400 for Missing Headers, got {response.status_code}"
    data = response.get_json()
    assert "errors" in data, f"Expected errors from missing headers, got {data}"
    errors = data["errors"]
    assert (
        len(errors) == 1
    ), f"Expected a single error from missing headers, got {len(errors)}"
    error = errors[0]
    assert (
        error["title"] == "Missing Headers"
    ), f"Expected Missing header header, got {error['title']}"


@no_type_check
def test_post_status_validation_error(client: FlaskClient) -> None:
    response = client.post(
        "/api/status",
        json={"SHT": "not_a_bool"},
        headers={
            HEADERS.TIMESTAMP.value: dt2str(datetime.now(tz=UTC)),
            HEADERS.MACADDRESS.value: "00:1A:2B:3C:4D:5E",
            HEADERS.FIRMWAREVERSION.value: "1.0.0",
        },
    )

    assert (
        response.status_code == 422
    ), f"Expected 422 for Validation Error, got {response.status_code} :: {response.get_json()}"
    data = response.get_json()
    assert "errors" in data, f"Expected errors from validation error, got {data}"
    errors = data["errors"]
    assert (
        len(errors) == 1
    ), f"Expected a single error from validation error, got {len(errors)}"
    error = errors[0]
    assert (
        error["title"] == "Validation Error"
    ), f"Expected Validation Error header, got {error['title']}"


@no_type_check
def test_post_status_notfound_error(client: FlaskClient) -> None:
    response = client.post(
        "/api/status",
        json={
            "SHT": False,
            "BMP": False,
            "CAM": False,
            "WIFI": False,
        },
        headers={
            HEADERS.TIMESTAMP.value: dt2str(datetime.now(tz=UTC)),
            HEADERS.MACADDRESS.value: "00:1A:2B:3C:4D:5E",  # Station does not exist
            HEADERS.FIRMWAREVERSION.value: "1.0.0",
        },
    )

    assert (
        response.status_code == 404
    ), f"Expected 404 for Not Found Error, got {response.status_code} :: {response.get_json()}"
    data = response.get_json()
    assert "errors" in data, f"Expected errors from not found error, got {data}"
    errors = data["errors"]
    assert (
        len(errors) == 1
    ), f"Expected a single error from not found error, got {len(errors)}"
    error = errors[0]
    assert (
        error["title"] == "Not Found"
    ), f"Expected Not Found header, got {error['title']}"


@no_type_check
def test_register_station_successful(client: FlaskClient) -> None:
    response = client.post(
        "/api/register",
        json={
            "name": "station",
            "device_model": "ESP32",
            "camera_model": "DSLR",
            "firmware_version": "1.0.0",
            "altitude": 400,
            "latitude": 83.3323,
            "longitude": 82.5546,
        },
        headers={
            HEADERS.TIMESTAMP.value: dt2str(datetime.now(tz=UTC)),
            HEADERS.MACADDRESS.value: "00:1A:2B:3C:4D:5E",
            HEADERS.FIRMWAREVERSION.value: "1.0.0",
        },
    )

    assert (
        response.status_code == 200
    ), f"Operation failed: {response.status_code} :: {response.get_json()}"
    data = response.get_json()
    assert data.get("status", None), "status not returned in message"
    assert data["status"] == "success", f"Expected success status, got {data['status']}"


@no_type_check
def test_register_station_already_exists(client: FlaskClient) -> None:
    # First we ensure the station actually exists.
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=StationStatus(
            MAC="00:1A:2B:3C:4D:5E",
            timestamp=datetime.now(tz=UTC),
            SHT=False,
            BMP=False,
            CAM=False,
            WIFI=False,
        ),
    )

    StationService.insert(station=station)

    # No error during insertion means we're good.

    response = client.post(
        "/api/register",
        json={
            "name": "station",
            "device_model": "ESP32",
            "camera_model": "DSLR",
            "firmware_version": "1.0.0",
            "altitude": 400,
            "latitude": 83.3323,
            "longitude": 82.5546,
        },
        headers={
            HEADERS.TIMESTAMP.value: dt2str(datetime.now(tz=UTC)),
            HEADERS.MACADDRESS.value: station.MAC,
            HEADERS.FIRMWAREVERSION.value: "1.0.0",
        },
    )

    assert (
        response.status_code == 409
    ), f"Expected 409 for Already Exists, got {response.status_code} :: {response.get_json()}"
    data = response.get_json()
    assert "errors" in data, f"Expected errors from already exists, got {data}"
    errors = data["errors"]
    assert (
        len(errors) == 1
    ), f"Expected a single error from already exists, got {len(errors)}"
    error = errors[0]
    assert (
        error["title"] == "Already Exists"
    ), f"Expected Already Exists header, got {error['title']}"


@no_type_check
def test_register_station_missing_headers(client: FlaskClient) -> None:
    response = client.post("/api/register", json={}, headers={})

    assert (
        response.status_code == 400
    ), f"Expected 400 for Missing Headers, got {response.status_code}"
    data = response.get_json()
    assert "errors" in data, f"Expected errors from missing headers, got {data}"
    errors = data["errors"]
    assert (
        len(errors) == 1
    ), f"Expected a single error from missing headers, got {len(errors)}"
    error = errors[0]
    assert (
        error["title"] == "Missing Headers"
    ), f"Expected Missing header header, got {error['title']}"


@no_type_check
def test_register_station_validation_error(client: FlaskClient) -> None:
    response = client.post(
        "/api/register",
        json={
            "name": "station",
            "device_model": "INVALID_DEVICE",  # Invalid device model
            "camera_model": "DSLR",
            "firmware_version": "1.0.0",
            "altitude": 400,
            "latitude": 83.3323,
            "longitude": 82.5546,
        },
        headers={
            HEADERS.TIMESTAMP.value: dt2str(datetime.now(tz=UTC)),
            HEADERS.MACADDRESS.value: "00:1A:2B:3C:4D:5E",
            HEADERS.FIRMWAREVERSION.value: "1.0.0",
        },
    )

    assert (
        response.status_code == 422
    ), f"Expected 422 for Validation Error, got {response.status_code} :: {response.get_json()}"
    data = response.get_json()
    assert "errors" in data, f"Expected errors from validation error, got {data}"
    errors = data["errors"]
    assert (
        len(errors) == 1
    ), f"Expected a single error from validation error, got {len(errors)}"
    error = errors[0]
    assert (
        error["title"] == "Validation Error"
    ), f"Expected Validation Error header, got {error['title']}"
