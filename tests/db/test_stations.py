from typing import no_type_check
from flask.testing import FlaskClient
from server.db import CameraModel, DeviceType, Station, StationService


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
def test_list_stations_multiple(client: FlaskClient) -> None:
    station1 = Station(
        MAC="00:1A:2B:3C:4D:5E",
        name="station1",
        device_model=DeviceType.ESP32,
        camera_model=CameraModel.DSLR,
        firmware_version="1.0.0",
        altitude=400,
        latitude=83.3323,
        longitude=82.5546,
        sensors=None,
    )
    station2 = Station(
        MAC="00:1A:2B:3C:4D:5F",
        name="station2",
        device_model=DeviceType.ESP8266,
        camera_model=CameraModel.OV5640,
        firmware_version="1.0.1",
        altitude=500,
        latitude=84.3323,
        longitude=83.5546,
        sensors=None,
    )
    StationService.insert(station=station1)
    StationService.insert(station=station2)

    stations = StationService.list()
    assert len(stations) == 2, "Expected at least two stations in the database"


@no_type_check
def test_list_station_offset(client: FlaskClient) -> None:
    stations = [
        Station(
            MAC=f"00:1A:2B:3C:4D:{i:02X}",
            name=f"station{i}",
            device_model=DeviceType.ESP32,
            camera_model=CameraModel.DSLR,
            firmware_version="1.0.0",
            altitude=400 + i * 10,
            latitude=83.3323 + i * 0.01,
            longitude=82.5546 + i * 0.01,
            sensors=None,
        )
        for i in range(10)
    ]

    for station in stations:
        StationService.insert(station=station)

    # Test with offset
    limit = 5
    page = 1
    offset = limit * page
    limited_stations = StationService.list(limit=limit, page=page)
    assert isinstance(limited_stations, list), "Expected a list of stations"
    assert (
        len(limited_stations) == len(stations) - offset
    ), f"Expected {len(stations) - offset} stations, got {len(limited_stations)}"


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


@no_type_check
def test_delete_station(client: FlaskClient) -> None:
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
    tempsta = StationService.get(MAC=station.MAC)
    assert tempsta is not None, "Station to be deleted could not be retrieved"
    assert tempsta == station, "Retrieved station does not match inserted station"

    # Delete the station
    StationService.delete(MAC=station.MAC)

    # Try to retrieve the deleted station
    deleted_station = StationService.get(MAC=station.MAC)
    assert deleted_station is None, "Deleted station should not be retrievable"


@no_type_check
def test_station_exists(client: FlaskClient) -> None:
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

    exists = StationService.exists(MAC=station.MAC)
    assert exists, "Station should exist after insertion"

    # Delete the station
    StationService.delete(MAC=station.MAC)

    exists_after_deletion = StationService.exists(MAC=station.MAC)
    assert not exists_after_deletion, "Station should not exist after deletion"
