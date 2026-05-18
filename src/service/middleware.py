import uuid
import time
import logging
from fastapi import Request
from typing import Callable, Awaitable
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from src.obs.metrics import http_requests_total, http_request_duration_seconds

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        request.state.latency_ms = elapsed_ms

        http_requests_total.labels(
            route=request.url.path,
            method=request.method,
            status=str(response.status_code),
        ).inc()

        elapsed_seconds = elapsed_ms / 1000

        http_request_duration_seconds.labels(
            route=request.url.path, method=request.method
        ).observe(elapsed_seconds)
        logger.info(
            "request",
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "route": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "model_version": None,
                "index_version": None,
                "batch_size": None,
                "seq_len_min": None,
                "seq_len_mean": None,
                "seq_len_max": None,
                "error_code": None,
            },
        )

        return response
