from typing import no_type_check
from flask.testing import FlaskClient
from server.db import (
    StationStatus,
    Station,
    StationStatusJSON,
    StatusService,
    StationService,
)
from datetime import datetime, UTC
from server.db import CameraModel, DeviceType


@no_type_check
def test_get_status_not_found(client: FlaskClient) -> None:
    result = StatusService.get(MAC="00:1A:2B:3C:4D:5E")
    assert (
        result is None,
        f"MAC address is not registered, should be None but got {result}",
    )


@no_type_check
def test_insert_status_success(client: FlaskClient) -> None:
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

    status = StatusService.get(MAC=station.MAC)
    assert status is not None, "Could not retrieve newly inserted status"
    assert isinstance(status, StationStatus), "Expected StationStatus instance"
    assert station.sensors == status, "Inserted status does not match retrieved status"


@no_type_check
def test_list_statuses(client: FlaskClient) -> None:
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

    statuses = StatusService.list()
    assert isinstance(statuses, list), "Expected a list of statuses"
    assert len(statuses) > 0, "Expected at least one status in the database"
    for status in statuses:
        assert isinstance(
            status, StationStatus
        ), f"Expected StationStatus instance, got {type(status)}"


@no_type_check
def test_list_statuses_multiple(client: FlaskClient) -> None:
    for i in range(5):
        station = Station(
            MAC=f"00:1A:2B:3C:4D:{i}E",
            name=f"station{i}",
            device_model=DeviceType.ESP32,
            camera_model=CameraModel.DSLR,
            firmware_version="1.0.0",
            altitude=400,
            latitude=83.3323,
            longitude=82.5546,
            sensors=StationStatus(
                MAC=f"00:1A:2B:3C:4D:{i}E",
                timestamp=datetime.now(tz=UTC),
                SHT=False,
                BMP=False,
                CAM=False,
                WIFI=False,
            ),
        )
        StationService.insert(station=station)

    statuses = StatusService.list()
    assert isinstance(statuses, list), "Expected a list of statuses"
    assert len(statuses) > 0, "Expected at least one status in the database"
    for status in statuses:
        assert isinstance(
            status, StationStatus
        ), f"Expected StationStatus instance, got {type(status)}"


@no_type_check
def test_list_statuses_offset(client: FlaskClient) -> None:

    for i in range(10):
        station = Station(
            MAC=f"00:1A:2B:3C:4D:{i:02X}",
            name=f"station{i}",
            device_model=DeviceType.ESP32,
            camera_model=CameraModel.DSLR,
            firmware_version="1.0.0",
            altitude=400,
            latitude=83.3323,
            longitude=82.5546,
            sensors=StationStatus(
                MAC=f"00:1A:2B:3C:4D:{i:02X}",
                timestamp=datetime.now(tz=UTC),
                SHT=False,
                BMP=False,
                CAM=False,
                WIFI=False,
            ),
        )
        StationService.insert(station=station)

    limit = 5
    page = 1
    offset = limit * page
    offset_statuses = StatusService.list(limit=limit, page=page)
    assert (
        len(offset_statuses) == 10 - offset
    ), f"Expected 5 statuses after offset, got {len(offset_statuses)}"
    for status in offset_statuses:
        assert isinstance(
            status, StationStatus
        ), f"Expected StationStatus instance, got {type(status)}"


@no_type_check
def test_update_status_success(client: FlaskClient) -> None:
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

    updated_status = StationStatus(
        MAC="00:1A:2B:3C:4D:5E",
        timestamp=datetime.now(tz=UTC),
        SHT=True,
        BMP=True,
        CAM=True,
        WIFI=True,
    )
    StatusService.update(MAC=station.MAC, status=updated_status)

    status = StatusService.get(MAC=station.MAC)
    assert status is not None, "Could not retrieve updated status"
    assert status == updated_status, "Updated status does not match retrieved status"


@no_type_check
def test_delete_status(client: FlaskClient) -> None:
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

    StatusService.delete(MAC=station.MAC)

    status = StatusService.get(MAC=station.MAC)
    assert status is None, "Status should be deleted, but still exists"


@no_type_check
def test_status_exists(client: FlaskClient) -> None:
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

    exists = StatusService.exists(MAC=station.MAC)
    assert exists, "Status should exist for the inserted station"
