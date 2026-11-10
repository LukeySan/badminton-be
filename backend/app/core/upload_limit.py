"""
Reject too-large video uploads before the body is read.

FastAPI reads the whole multipart body (into a temp file) before the route runs,
so a size check in the route would come only after a 5 GB upload had finished.
This middleware looks at the Content-Length header first: too large → 413 at once.
Uploads without Content-Length get 411, since their size can't be known up front
(browsers always send it for a file). The route still counts the bytes it saves,
because Content-Length also includes the multipart wrapping around the file.
"""

import json
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import settings

# Content-Length covers the multipart wrapping (boundaries, headers, form fields)
# as well as the file, so allow a little more here; the route checks the file exactly.
MULTIPART_OVERHEAD = 64 * 1024


def format_bytes(n: float) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e8 else f"{n / 1e6:.1f} MB"


class UploadSizeLimit:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        upload_path = f"{settings.API_V1_STR}/videos"
        if (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"].rstrip("/") == upload_path
        ):
            headers = dict(scope["headers"])
            length = headers.get(b"content-length")
            limit = settings.MAX_UPLOAD_BYTES  # read per request (tests change it)
            if length is None:
                return await self._reply(
                    send, 411, "Uploads need a Content-Length header"
                )
            if not length.isdigit():
                return await self._reply(send, 400, "Invalid Content-Length header")
            if int(length) > limit + MULTIPART_OVERHEAD:
                return await self._reply(
                    send,
                    413,
                    f"The file is {format_bytes(int(length))}; "
                    f"the limit is {format_bytes(limit)}",
                )
        await self.app(scope, receive, send)

    @staticmethod
    async def _reply(send: Send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        headers: list[Any] = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            # The client is still sending the body we won't read
            (b"connection", b"close"),
        ]
        await send(
            {"type": "http.response.start", "status": status, "headers": headers}
        )
        await send({"type": "http.response.body", "body": body})
