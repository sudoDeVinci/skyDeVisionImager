import pytest
from typing import no_type_check
from flask.testing import FlaskClient
from server.db import CameraModel, DeviceType, Station, StationJSON, StationService


@no_type_check
def test_get_station_not_found(client: FlaskClient) -> None:
    result = StationService.get(MAC="00:1A:2B:3C:4D:5E")
    assert (
        result is None
    ), f"MAC address is not registered, should be None but got {result}"


@no_type_check
def test_insert_station_sucess(client: FlaskClient) -> None:
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    StationService.insert(station=station)
    newstation = StationService.get(MAC=station.MAC)
    assert newstation is not None, "Could not retrieve newly inserted station"
    assert newstation == station


@no_type_check
def test_list_stations(client: FlaskClient) -> None:
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    StationService.insert(station=station)
    stations = StationService.list()
    assert isinstance(stations, list), "Expected a list of stations"
    assert len(stations) > 0, "Expected at least one station in the database"
    for station in stations:
        assert isinstance(
            station, Station
        ), f"Expected Station instance, got {type(station)}"


@no_type_check
def test_update_station(client: FlaskClient) -> None:
    station = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    StationService.insert(station=station)

    # Update the station's name
    updates = {"name": "updated_station", "device_model": DeviceType.ESP8266}
    StationService.update(MAC=station.MAC, station=updates)

    updated_station = StationService.get(MAC=station.MAC)
    assert updated_station is not None, "Updated station could not be retrieved"
    assert (
        updated_station.name == "updated_station"
    ), "Station name was not updated correctly"
    assert (
        updated_station.device_model == DeviceType.ESP8266
    ), "Device model was not updated correctly"
