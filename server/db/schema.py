from sqlite3 import Cursor


def apply_schema(cursor: Cursor | None) -> None:
    if cursor is None:
        raise ValueError("Cursor is required")
    # Create the Stations table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Stations(
            MAC TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            device_model TEXT NOT NULL,
            camera_model TEXT NOT NULL,
            firmware_version TEXT NOT NULL,
            altitude REAL NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL
        );
    """
    )

    # Create the Readings table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Readings(
            MAC TEXT,
            timestamp TEXT,
            temperature REAL,
            humidity REAL,
            pressure REAL,
            dewpoint REAL,
            filepath TEXT,
            PRIMARY KEY (MAC, timestamp),
            FOREIGN KEY (MAC) REFERENCES Stations(MAC)
        );
    """
    )

    # Create the Status table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Status(
            MAC TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            SHT INTEGER NOT NULL,
            BMP INTEGER NOT NULL,
            CAM INTEGER NOT NULL,
            WIFI INTEGER NOT NULL,
            FOREIGN KEY (MAC) REFERENCES Stations(MAC)
        );
    """
    )

    # Create the Locations table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Locations(
            country TEXT,
            region TEXT,
            city TEXT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            PRIMARY KEY (latitude, longitude)
        );
    """
    )

    # Create the Users table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Users(
            ID TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        );
    """
    )
