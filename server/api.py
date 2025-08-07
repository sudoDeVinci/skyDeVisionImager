from sqlite3 import OperationalError as SQLError
from .db import (
    StatusService,
    StationService,
    DatabaseError,
    InternalDBError,
    InvalidInputError,
    NotFoundError,
    AlreadyExistsError
)

from flask import (
    Flask,
    Blueprint,
    Response,
    request,
    jsonify
)

from ._utils import (
    headercheck,
    HEADERS as HEADERS,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
)

from ._types import ErrorResponse, ErrorDict


apiRouter = Blueprint('api', __name__, url_prefix='/api')
"""
API endpoints for handling status and environmental reading data.
"""

@apiRouter.route('/status', methods=['POST', 'PUT'])
def status():
    """
    Endpoint to handle status updates.
    """

    try:
        headers = request.headers
        headercheck(headers)
        mac = headers.get(HEADERS.MACADDRESS)
        timestamp = headers.get(HEADERS.TIMESTAMP)
        statusdict = request.get_json()

        StatusService.update(MAC=mac, status=statusdict, timestamp=timestamp)

    except ValueError as ve:
        return ErrorResponse(
            errors=[
                ErrorDict(
                    title="Invalid ",
                    details=str(ve),
                    code="400",
                    source="status"
                )
            ],
            timestamp=request.headers.get(HEADERS.TIMESTAMP, "unknown")
        )

@apiRouter.route('/reading', methods=['POST', 'PUT'])
def reading():
    """
    Endpoint to handle environmental reading updates.
    """
    pass

@apiRouter.route('/qnh', methods=['GET'])
def qnh():
    """
    Endpoint to handle QNH updates.
    """
    pass


@apiRouter.route('/version', methods=['GET'])
def version():
    """
    Endpoint to handle firmware version updates.
    """
    pass


@apiRouter.route('/check', methods=['GET'])
def index():
    pass







