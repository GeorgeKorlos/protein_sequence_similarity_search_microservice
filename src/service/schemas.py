from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    sequences: list[str] = Field(min_length=1)


class SearchRequest(
    BaseModel,
):
    sequence: str
    top_k: int = Field(default=10, le=50, ge=1)  # default 10, max 50


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model_version: str
    request_id: str


class SearchResult(BaseModel):
    rank: int
    accession: str
    score: float
    organism: str
    keywords: list[str]
    go_terms: list[str]


class SearchResponse(BaseModel):
    query_length: int
    results: list[SearchResult]
    model_version: str
    index_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
    service_version: str
    model_version: str
    corpus_version: str
    index_version: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    index_loaded: bool


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str
