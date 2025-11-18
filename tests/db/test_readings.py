from datetime import datetime, UTC, timedelta
from typing import no_type_check
from flask.testing import FlaskClient
from server.db import (
    CameraModel,
    DeviceType,
    Station,
    StationService,
    Reading,
    ReadingService,
    NotFoundError,
)


@no_type_check
def test_get_reading_not_found(client: FlaskClient) -> None:
    result = ReadingService.get(MAC="00:1A:2B:3C:4D:5E", timestamp=datetime.now())
    assert result is None, (
        f"MAC address is not registered, should be None but got {result}"
    )


@no_type_check
def test_insert_and_get_reading(client: FlaskClient) -> None:
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

    dt = datetime.now(tz=UTC)
    reading = Reading(
        MAC=station.MAC,
        timestamp=dt,
        temperature=0.0,
        humidity=0.0,
        pressure=0.0,
        dewpoint=0.0,
    )
    ReadingService.insert(reading=reading)

    newreading = ReadingService.get(MAC=reading.MAC, timestamp=reading.timestamp)

    assert newreading is not None, "Could not retrieve inserted reading."
    assert isinstance(newreading, Reading), (
        f"Reading hsould be of type 'Reading', got {type(newreading)}"
    )

    assert newreading == reading, "Inserted reading is not equivalent to extracted"


@no_type_check
def test_list_readings(client: FlaskClient) -> None:
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

    dt = datetime.now(tz=UTC)
    reading = Reading(
        MAC=station.MAC,
        timestamp=dt,
        temperature=0.0,
        humidity=0.0,
        pressure=0.0,
        dewpoint=0.0,
    )
    ReadingService.insert(reading=reading)

    readings = ReadingService.list(MAC=station.MAC)
    assert isinstance(readings, list), (
        f"Expected readings to be list, got {type(readings)}"
    )
    assert len(readings) == 1, f"Readings should be length 1, got {len(readings)}"
    assert readings[0] == reading, "Inserted reading is not equivalent to extracted"


@no_type_check
def test_list_readings_startdate(client: FlaskClient) -> None:
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

    startdatetime = datetime.now(tz=UTC)
    dtinvalid = startdatetime - timedelta(hours=1)
    dtvalid = startdatetime + timedelta(hours=1)

    readingvalid = Reading(
        MAC=station.MAC,
        timestamp=dtvalid,
        temperature=0.0,
        humidity=0.0,
        pressure=0.0,
        dewpoint=0.0,
    )

    readinginvalid = Reading(
        MAC=station.MAC,
        timestamp=dtinvalid,
        temperature=0.1,
        humidity=0.1,
        pressure=0.1,
        dewpoint=0.1,
    )

    ReadingService.insert(reading=readinginvalid)
    ReadingService.insert(reading=readingvalid)

    readings = ReadingService.list(MAC=station.MAC, start=startdatetime)
    assert isinstance(readings, list), (
        f"Expected readings to be list, got {type(readings)}"
    )
    assert len(readings) == 1, f"Readings should be length 1, got {len(readings)}"
    assert readings[0] == readingvalid, (
        "Inserted reading is not equivalent to extracted"
    )


@no_type_check
def test_list_readings_enddate(client: FlaskClient) -> None:
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

    enddatetime = datetime.now(tz=UTC)
    dtinvalid = enddatetime + timedelta(hours=1)
    dtvalid = enddatetime - timedelta(hours=1)

    readingvalid = Reading(
        MAC=station.MAC,
        timestamp=dtvalid,
        temperature=0.0,
        humidity=0.0,
        pressure=0.0,
        dewpoint=0.0,
    )

    readinginvalid = Reading(
        MAC=station.MAC,
        timestamp=dtinvalid,
        temperature=0.1,
        humidity=0.1,
        pressure=0.1,
        dewpoint=0.1,
    )

    ReadingService.insert(reading=readinginvalid)
    ReadingService.insert(reading=readingvalid)

    readings = ReadingService.list(MAC=station.MAC, end=enddatetime)
    assert isinstance(readings, list), (
        f"Expected readings to be list, got {type(readings)}"
    )
    assert len(readings) == 1, f"Readings should be length 1, got {len(readings)}"
    assert readings[0] == readingvalid, (
        "Inserted reading is not equivalent to extracted"
    )


@no_type_check
def test_list_readings_startdate_and_enddate(client: FlaskClient) -> None:
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

    startdatetime = datetime.now(tz=UTC)
    dtvalid = startdatetime + timedelta(hours=1)
    enddatetime = startdatetime + timedelta(hours=2)
    dtinvalid = enddatetime + timedelta(hours=1)

    readingvalid = Reading(
        MAC=station.MAC,
        timestamp=dtvalid,
        temperature=0.0,
        humidity=0.0,
        pressure=0.0,
        dewpoint=0.0,
    )

    readinginvalid = Reading(
        MAC=station.MAC,
        timestamp=dtinvalid,
        temperature=0.1,
        humidity=0.1,
        pressure=0.1,
        dewpoint=0.1,
    )

    ReadingService.insert(reading=readinginvalid)
    ReadingService.insert(reading=readingvalid)

    readings = ReadingService.list(
        MAC=station.MAC, start=startdatetime, end=enddatetime
    )
    assert isinstance(readings, list), (
        f"Expected readings to be list, got {type(readings)}"
    )
    assert len(readings) == 1, f"Readings should be length 1, got {len(readings)}"
    assert readings[0] == readingvalid, (
        "Inserted reading is not equivalent to extracted"
    )


@no_type_check
def test_list_readings_multiple(client: FlaskClient) -> None:
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

    dt = datetime.now(tz=UTC)
    readings = [
        Reading(
            MAC=station.MAC,
            timestamp=dt + timedelta(minutes=i),
            temperature=float(i),
            humidity=float(i),
            pressure=float(i),
            dewpoint=float(i),
        )
        for i in range(5)
    ]

    for reading in readings:
        ReadingService.insert(reading=reading)

    newreadings = ReadingService.list(MAC=station.MAC)
    newreadings.sort(key=lambda r: r.timestamp)

    assert isinstance(readings, list), (
        f"Expected readings to be list, got {type(readings)}"
    )
    assert len(readings) == 5, f"Readings should be length 5, got {len(readings)}"
    for i in range(5):
        assert newreadings[i] == readings[i], (
            "Inserted reading is not equivalent to extracted"
        )


@no_type_check
def test_update_readings_successful(client: FlaskClient) -> None:
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

    dt = datetime.now(tz=UTC)
    reading = Reading(
        MAC=station.MAC,
        timestamp=dt,
        temperature=0.0,
        humidity=0.0,
        pressure=0.0,
        dewpoint=0.0,
    )
    ReadingService.insert(reading=reading)

    updates = {"temperature": 0.5, "humidity": 0.5, "pressure": 0.5, "dewpoint": 0.5}
    updatedreading = reading.model_copy(update=updates, deep=True)

    ReadingService.update(MAC=station.MAC, timestamp=dt, reading=updatedreading)

    fetchedreading = ReadingService.get(MAC=station.MAC, timestamp=dt)
    if fetchedreading is None:
        raise NotFoundError("Updated reading could not be found.")
    assert isinstance(fetchedreading, Reading), (
        f"Reading hsould be of type 'Reading', got {type(fetchedreading)}"
    )

    assert fetchedreading == updatedreading, (
        "Updated reading is not equivalent to extracted"
    )


@no_type_check
def test_update_readings_nonexistent(client: FlaskClient) -> None:
    reading = Reading(
        MAC="00:1A:2B:3C:4D:5E",
        timestamp=datetime.now(tz=UTC),
        temperature=0.0,
        humidity=0.0,
        pressure=0.0,
        dewpoint=0.0,
    )

    try:
        ReadingService.update(
            MAC=reading.MAC, timestamp=reading.timestamp, reading=reading
        )
        raise AssertionError(
            "Updating a non-existent reading should raise NotFoundError"
        )
    except NotFoundError:
        pass


@no_type_check
def test_delete_reading_successful(client: FlaskClient) -> None:
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

    dt = datetime.now(tz=UTC)
    reading = Reading(
        MAC=station.MAC,
        timestamp=dt,
        temperature=0.0,
        humidity=0.0,
        pressure=0.0,
        dewpoint=0.0,
    )
    ReadingService.insert(reading=reading)

    fetchedreading = ReadingService.get(MAC=station.MAC, timestamp=dt)
    if fetchedreading is None:
        raise NotFoundError("Inserted reading could not be found.")
    assert isinstance(fetchedreading, Reading), (
        f"Reading hsould be of type 'Reading', got {type(fetchedreading)}"
    )

    assert fetchedreading == reading, "Inserted reading is not equivalent to extracted"

    ReadingService.delete(MAC=station.MAC, timestamp=dt)
    deletedreading = ReadingService.get(MAC=station.MAC, timestamp=dt)
    assert deletedreading is None, (
        f"Deleted reading should be None, got {deletedreading}"
    )


@no_type_check
def test_delete_reading_nonexistent(client: FlaskClient) -> None:
    try:
        ReadingService.delete(MAC="00:1A:2B:3C:4D:5E", timestamp=datetime.now(tz=UTC))
        raise AssertionError(
            "Deleting a non-existent reading should raise NotFoundError"
        )
    except NotFoundError:
        pass


@no_type_check
def test_exist_reading_successful(client: FlaskClient) -> None:
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

    dt = datetime.now(tz=UTC)
    reading = Reading(
        MAC=station.MAC,
        timestamp=dt,
        temperature=0.0,
        humidity=0.0,
        pressure=0.0,
        dewpoint=0.0,
    )
    ReadingService.insert(reading=reading)

    exists = ReadingService.exists(MAC=station.MAC, timestamp=dt)
    assert exists is True, f"Reading should exist, got {exists}"


@no_type_check
def test_exists_reading_nonexistent(client: FlaskClient) -> None:
    exists = ReadingService.exists(
        MAC="00:1A:2B:3C:4D:5E", timestamp=datetime.now(tz=UTC)
    )
    assert exists is False, f"Reading should not exist, got {exists}"
