"""
Errors with a machine-readable code next to the message:

    {"detail": "No shuttle trajectory for this video...", "code": "trajectory_missing"}

For the few errors the app reacts to with something other than showing the message
(e.g. offering "Re-analyze"). Matching on the code, not the message text, lets the
message change freely. Every other error stays a plain {"detail": ...}.
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from app.models import ErrorCode, ErrorWithCode


class CodedHTTPException(HTTPException):
    def __init__(self, status_code: int, code: ErrorCode, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.code = code


async def coded_http_exception_handler(
    request: Request,  # noqa: ARG001
    exc: Exception,
) -> JSONResponse:
    assert isinstance(exc, CodedHTTPException)
    body = ErrorWithCode(detail=exc.detail, code=exc.code)
    return JSONResponse(body.model_dump(mode="json"), status_code=exc.status_code)
