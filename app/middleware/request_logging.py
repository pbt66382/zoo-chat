"""
为每个请求分配 X-Request-ID，记录访问日志与未捕获异常栈。
"""
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("zoo_chat.http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = rid

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed request_id=%s method=%s path=%s",
                rid,
                request.method,
                request.url.path,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request_done request_id=%s method=%s path=%s status=%s elapsed_ms=%.2f",
            rid,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        response.headers["X-Request-ID"] = rid
        return response
