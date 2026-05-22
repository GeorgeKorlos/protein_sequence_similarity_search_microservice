import logging
import platform
from fastapi import FastAPI, Request
from src.service.routes import router
from src.core.searcher import Searcher
from src.service.config import settings
from src.obs.metrics import errors_total
from contextlib import asynccontextmanager
from src.core.embedder import ESM2Embedder
from fastapi.responses import JSONResponse
from src.service.schemas import ErrorResponse
from src.core.corpus_store import CorpusStore
from src.service.config import SERVICE_VERSION
from src.obs.metrics import service_build_info
from src.core.index_manager import IndexManager
from src.obs.logging_config import setup_logging
from src.core.validator import SequenceValidator
from src.core.exceptions import ServiceException
from src.service.middleware import TimingMiddleware, RequestIDMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    setup_logging()

    app.state.settings = settings

    app.state.corpus = CorpusStore(csv_path=settings.corpus_path)

    app.state.embedder = ESM2Embedder(
        model_tag=settings.model_tag, device=settings.device, compile_model=False
    )
    try:
        app.state.embedder.embed(["ACDE"])
        app.state.model_loaded = True
    except Exception as e:
        app.state.model_loaded = False
        logger.error("Model failed to load: %s", e, exc_info=True)

    app.state.index_manager = IndexManager(index_type="", dim=0, params=None)
    try:
        app.state.index_manager.load(settings.index_path)
        if app.state.index_manager.index.ntotal > 0:  # type: ignore
            app.state.index_loaded = True
        else:
            app.state.index_loaded = False
            logger.error("Index not loaded")
    except Exception as e:
        app.state.index_loaded = False
        logger.error("Index failed to load %s", e)
    if app.state.model_loaded and app.state.index_loaded:
        service_build_info.info(
            {
                "service_version": str(SERVICE_VERSION),
                "model_version": str(app.state.embedder.model_version),
                "index_version": str(app.state.index_manager.index_version["checksum"]),
                "python_version": str(platform.python_version()),
            }
        )
    app.state.validator = SequenceValidator(
        max_batch_size=settings.max_batch_size,
        max_payload_size=settings.max_payload_size,
    )
    try:
        app.state.searcher = Searcher(
            corpus_store=app.state.corpus,
            embedder=app.state.embedder,
            index_manager=app.state.index_manager,
        )
    except Exception as e:
        app.state.searcher = None
        logger.error("Searcher failed to initialize: %s", e)

    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(TimingMiddleware)  # type: ignore
app.add_middleware(RequestIDMiddleware)  # type: ignore


@app.exception_handler(ServiceException)
async def service_exception_handler(
    request: Request, exc: ServiceException
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    error_response = ErrorResponse(
        error_code=exc.error_code, message=exc.message, request_id=request_id
    )
    errors_total.labels(error_code=exc.error_code).inc()

    return JSONResponse(
        status_code=exc.http_status, content=error_response.model_dump()
    )


app.include_router(router=router)
