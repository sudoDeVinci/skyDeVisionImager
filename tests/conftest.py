from collections.abc import Generator
import pytest
import tempfile
import os
from server import create_app, Manager
from flask import Flask, testing


@pytest.fixture(autouse=True)
def mock_app() -> Generator[Flask, None, None]:
    app: Flask = create_app()
    app.config.update(
        {
            "TESTING": True,
        }
    )
    yield app


@pytest.fixture()
def client(mock_app: Flask) -> Generator[testing.FlaskClient, None, None]:
    # Create a temporary database file that gets deleted after the test
    with tempfile.NamedTemporaryFile(suffix=".db", delete=True) as temp_db:
        temp_db_path = temp_db.name

        # Close the file so SQLite can open it
        temp_db.close()

        try:
            Manager.connect(temp_db_path, uri=False)
            yield mock_app.test_client()
        finally:
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
