import time
import asyncio
import logging
from src.core.searcher import Searcher
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import JSONResponse, Response
from src.core.embedder import ESM2Embedder
from src.core.validator import SequenceValidator
from fastapi import APIRouter, Request, Depends
from src.obs.metrics import (
    model_inference_seconds,
    embed_batch_size,
    faiss_search_seconds,
)
from src.core.exceptions import (
    InvalidSequenceException,
    EmbeddingFailedException,
    SearchFailedException,
    SequenceValidationError,
    SequenceTooLongException,
    BatchTooLargeException,
    PayloadTooLargeException,
    SEQUENCE_TOO_LONG,
    BATCH_TOO_LARGE,
    PAYLOAD_TOO_LARGE,
    IndexNotReadyException,
)
from src.service.schemas import (
    EmbedRequest,
    SearchRequest,
    EmbedResponse,
    SearchResponse,
    SearchResult,
    HealthResponse,
    ReadyResponse,
)
from src.service.dependencies import (
    get_embedder,
    get_searcher,
    get_validator,
)
from src.service.config import SERVICE_VERSION

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
async def embed(
    request: Request,
    payload: EmbedRequest,
    embedder: ESM2Embedder = Depends(get_embedder),
    validator: SequenceValidator = Depends(get_validator),
) -> EmbedResponse:
    try:
        validated_sequences = validator.validate_batch(
            payload.sequences,
        )

    except SequenceValidationError as exc:
        if exc.error_code == SEQUENCE_TOO_LONG:
            raise SequenceTooLongException(str(exc)) from exc
        elif exc.error_code == BATCH_TOO_LARGE:
            raise BatchTooLargeException(str(exc)) from exc
        elif exc.error_code == PAYLOAD_TOO_LARGE:
            raise PayloadTooLargeException(str(exc)) from exc
        else:
            raise InvalidSequenceException(str(exc)) from exc

    try:
        embed_start = time.perf_counter()
        embeddings = await asyncio.to_thread(
            embedder.embed,
            validated_sequences,
        )
        end = time.perf_counter() - embed_start

        model_inference_seconds.observe(end)
        embed_batch_size.observe(len(validated_sequences))

    except Exception as exc:
        raise EmbeddingFailedException(
            "Failed to generate embeddings.",
        ) from exc

    return EmbedResponse(
        embeddings=embeddings.tolist(),
        model_version=embedder.model_version,
        request_id=request.state.request_id,
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: Request,
    payload: SearchRequest,
    searcher: Searcher = Depends(get_searcher),
    validator: SequenceValidator = Depends(get_validator),
) -> SearchResponse:
    try:
        validated_sequence = validator.validate(payload.sequence)

    except SequenceValidationError as exc:
        raise InvalidSequenceException(str(exc)) from exc

    if not request.app.state.index_loaded:
        raise IndexNotReadyException("Index is not loaded.")

    try:
        search_start = time.perf_counter()
        search_results = await asyncio.to_thread(
            searcher.search, validated_sequence, payload.top_k
        )
        end = time.perf_counter() - search_start

        faiss_search_seconds.observe(end)

    except Exception as exc:
        logger.error("Search failed: %s", exc, exc_info=True)
        raise SearchFailedException("Failed to search.") from exc

    return SearchResponse(
        query_length=len(validated_sequence),
        results=[SearchResult(**r) for r in search_results],
        model_version=request.app.state.embedder.model_version,
        index_version=request.app.state.index_manager.index_version,
        request_id=request.state.request_id,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service_version=SERVICE_VERSION,
        model_version=request.app.state.embedder.model_version,
        corpus_version=request.app.state.corpus.corpus_version,
        index_version=request.app.state.index_manager.index_version,
    )


@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request):
    model_loaded = request.app.state.model_loaded
    index_loaded = request.app.state.index_loaded

    ready = model_loaded and index_loaded

    response = ReadyResponse(
        ready=ready,
        model_loaded=model_loaded,
        index_loaded=index_loaded,
    )

    if not ready:
        return JSONResponse(
            status_code=503,
            content=response.model_dump(),
        )

    return response


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
