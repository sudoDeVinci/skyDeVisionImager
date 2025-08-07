from queue import Queue
from sqlite3 import connect, Connection, Cursor, Error as SQLError
from threading import Lock
from typing import Optional, Generator
from pathlib import Path
import json
from logging import INFO, FileHandler, Logger, StreamHandler, basicConfig, getLogger, ERROR, DEBUG
from contextlib import contextmanager
from .schema import apply_schema
from dotenv import load_dotenv
from os import environ
from bcrypt import hashpw, gensalt, checkpw


class SQLiteConnectionPool:
    """
    A thread-safe connection pool for SQLite database connections.

    Attributes:
        size (int): Maximum number of connections in the pool
        timeout (float): Timeout in seconds for getting a connection
        database (str): Path to the SQLite database file
    """

    def __init__(
        self, database: str, size: int = 5, timeout: float = 30.0, uri: bool = False
    ) -> None:
        self.database = database
        self.size = size
        self.timeout = timeout
        self._pool: Queue[Connection] = Queue(maxsize=size)
        self._lock = Lock()
        self.uri = uri
        self._initialize_pool()

    def _initialize_pool(self) -> None:
        """Initialize the connection pool with the specified number of connections."""
        for _ in range(self.size):
            conn = connect(
                database=self.database,
                timeout=self.timeout,
                check_same_thread=False,  # Required for multi-threaded access
                uri=self.uri,
            )
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 30000")
            self._pool.put(conn)

    def get_connection(self) -> Optional[Connection]:
        """Get a connection from the pool."""
        try:
            return self._pool.get(timeout=self.timeout)
        except Exception as e:
            print(f"Error getting connection from pool: {e}")
            return None

    def return_connection(self, connection: Connection) -> None:
        """Return a connection to the pool."""
        try:
            self._pool.put(connection, timeout=self.timeout)
        except Exception as e:
            print(f"Error returning connection to pool: {e}")
            connection.close()

    def closeall(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            conn = self._pool.get_nowait()
            conn.close()


class Manager:
    """
    Static Management class for Database configuration.
    """

    _pool: Optional[SQLiteConnectionPool] = None
    _configfile: Path = Path("configs") / "config.json"
    _dbfile: Optional[Path] = None
    _default_dbfile: Path = Path("database.db")
    _logfile: Path = Path("logs") / "server.db.log"
    _secret: Optional[str] = None
    logger: Logger

    @classmethod
    def set_database_file(cls, db_path: str | Path) -> None:
        """
        Set the database file path to use.

        Args:
            db_path: Path to the database file (string or Path object)
        """
        cls._dbfile = Path(db_path)
        cls.log(f"Database file set to: {cls._dbfile}")

    @classmethod
    def get_database_file(cls) -> Path:
        """
        Get the current database file path.

        Returns:
            Path: Current database file path
        """
        if cls._dbfile is None:
            cls._dbfile = cls._default_dbfile
        return cls._dbfile

    @classmethod
    def log(cls, message: str, level: int = INFO) -> None:
        """Log a message to the logger."""
        if not hasattr(cls, "logger"):
            cls.load()
        cls.logger.log(level, message)

    @classmethod
    def load(cls) -> None:
        """Load the configuration and logger objects."""
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("configs").mkdir(exist_ok=True)

        basicConfig(
            level=INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                StreamHandler(),
                FileHandler(str(cls._logfile)),
            ],
        )
        cls.logger = getLogger(__name__)
        cls.logger.setLevel(INFO)
        load_dotenv()
        cls._secret = environ.get("DBSECRET", None)

        if cls._secret is None:
            cls.log("DBSECRET not found in environment variables. Cannot continue.", level=INFO)
            raise ValueError("DBSECRET not found in environment variables.")
        
        try:
            with open(cls._configfile, "r") as file:
                data = json.load(file)
                # TODO: Load configuration data
                if not data:
                    raise json.JSONDecodeError("Empty file", str(cls._configfile), 0)
        except FileNotFoundError as err:
            cls.log(f"Error loading configuration: {err}")
        except json.JSONDecodeError as err:
            cls.log(f"Error parsing configuration: {err}")

    def hash_password(cls, password: str) -> str:
        """
        Hash + salt a password via bcrypt using the database secret key.

        Args:
            - password(str): The password to hash

        Returns:
            - str: The hashed password
        """

        hashed: bytes = b""

        try:
            peppered = f"{password}{cls._secret}"
            hashed = hashpw(peppered.encode("utf-8"), gensalt())
        except Exception as err:
            cls.log(f"Error hashing password::: {err}", level=ERROR)
            hashed = hashpw(password.encode("utf-8"), gensalt())

        return hashed.decode("utf-8")


    def password_match(cls, stored_hash: str, provided_password: str) -> bool:
        """
        Check if a password matches the stored hash.

        Args:
            - stored_hash(str): The stored password hash
            - provided_password(str): The password to check

        Returns:
            - bool: True if the password matches, False otherwise
        """
        try:
            peppered = f"{provided_password}{cls._secret}"
            return checkpw(peppered.encode("utf-8"), stored_hash.encode("utf-8"))
        except Exception as err:
            cls.log(f"Error checking password::: {err}", level=ERROR)
            return checkpw(provided_password.encode("utf-8"), stored_hash.encode("utf-8"))


    @classmethod
    def connected(cls) -> bool:
        """Check if the database is connected."""
        return cls._pool is not None

    @classmethod
    def connect(cls, db_path: Optional[str | Path] = None, uri: bool = False) -> None:
        """
        Connect to the SQLite database using the connection pool.

        Args:
            db_path: Optional path to the database file. If not provided, uses the default or previously set path.
        """
        cls.load()

        # Set database file if provided
        if db_path is not None:
            cls.set_database_file(db_path)

        try:
            current_db_path = cls.get_database_file()
            cls.log(f"Connecting to database: {current_db_path}")

            cls._pool = SQLiteConnectionPool(
                database=str(current_db_path),
                size=5,  # Adjust pool size as needed
                timeout=10.0,
                uri=uri,
            )

            with cls.cursor() as cursor:
                if cursor is None:
                    raise ValueError("Cursor is required")
                apply_schema(cursor)
                cursor.connection.commit()
        except SQLError as err:
            cls._pool = None
            cls.log(f"Error connecting to the database: {err}")

    @classmethod
    @contextmanager
    def connection(cls) -> Generator[Optional[Connection], None, None]:
        """
        Get a connection from the pool. Can be used as a context manager.

        Returns:
        --------
            Connection: Connection object from the SQLite database pool.

        Usage:
        ------
            # As a context manager:
            with Manager.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")

            # Or traditional way:
            conn = Manager.connection()
            try:
                # use connection
                pass
            finally:
                if conn and cls._pool:
                    cls._pool.return_connection(conn)
        """
        if not cls.connected():
            cls.connect()

        conn = cls._pool.get_connection() if cls._pool else None
        try:
            yield conn
        finally:
            if conn and cls._pool:
                cls._pool.return_connection(conn)

    @classmethod
    @contextmanager
    def cursor(cls) -> Generator[Optional[Cursor], None, None]:
        """
        Get a cursor from a pooled connection. Can be used as a context manager.

        Returns:
        --------
            Cursor: Cursor object from the SQLite database connection.

        Usage:
        ------
            # As a context manager:
            with Manager.cursor() as cursor:
                cursor.execute("SELECT * FROM table")
                results = cursor.fetchall()

            # Or traditional way:
            cursor = Manager.cursor()
            try:
                # use cursor
                pass
            finally:
                if cursor and cursor.connection and cls._pool:
                    cls._pool.return_connection(cursor.connection)
        """
        with cls.connection() as conn:
            if conn:
                cursor = conn.cursor()
                try:
                    yield cursor
                    conn.commit()  # Auto-commit successful transactions
                except Exception:
                    conn.rollback()  # Rollback on error
                    raise
            else:
                yield None

    @classmethod
    def close(cls) -> None:
        """Close all connections in the pool."""
        if cls._pool:
            cls._pool.closeall()
            cls._pool = None
