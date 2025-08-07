from flask import (
    Flask,
    Blueprint,
    Response,
    request,
    jsonify
)

apiRouter = Blueprint('api', __name__, url_prefix='/api')
"""
API endpoints for handling status and environmental reading data.
"""

@apiRouter.route('/status', methods=['POST', 'PUT'])
def status():
    """
    Endpoint to handle status updates.
    """
    pass

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







