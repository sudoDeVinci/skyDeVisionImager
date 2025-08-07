from typing import TypedDict, Any

class ErrorDict(TypedDict, total=False):
    """
    Schema for error responses.
    The exct meaning for each field are loose
    and are not strictly defined.
    This is mostly intended to mirror the structure
    of the error responses returned by the Ebanq API.
    """

    title: str
    details: str
    code: str
    source: str
    target: str
    meta: dict[str, Any]


class ErrorResponse(TypedDict, total=False):
    """
    Errors and warning response schema.
    This mirrors the structure of the error responses
    returned by the Ebanq API.
    It contains a list of errors and warnings encountered during the request.
    The `timestamp` field is used to indicate when the response was generated.
    """

    errors: list[ErrorDict]
    warnings: list[ErrorDict]
    timestamp: str
