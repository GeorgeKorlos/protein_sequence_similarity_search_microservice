import pytest
from pydantic import ValidationError
from src.service.schemas import EmbedRequest, SearchRequest, EmbedResponse


def test_valid_embed_request_with_single_sequence_parses_correctly():
    req = EmbedRequest(sequences=["ACDE"])
    assert req.sequences == ["ACDE"]


def test_embed_request_with_empty_sequences_list_fails_validation():
    with pytest.raises(ValidationError):
        EmbedRequest(sequences=[])


def test_search_request_missing_sequence_field_fails_validation():
    with pytest.raises(ValidationError):
        SearchRequest(top_k=10)  # type: ignore


def test_search_request_with_top_k_greater_than_maximum_fails_validation():
    with pytest.raises(ValidationError):
        SearchRequest(sequence="ACDEFGHIK", top_k=51)


def test_search_request_with_top_k_equal_to_zero_fails_validation():
    with pytest.raises(ValidationError):
        SearchRequest(sequence="ACDEFGHIK", top_k=0)


def test_embed_response_with_flat_embeddings_list_fails_validation():
    with pytest.raises(ValidationError):
        EmbedResponse(embeddings=[0.1, 0.2, 0.3], model_version="v1", request_id="123")  # type: ignore
