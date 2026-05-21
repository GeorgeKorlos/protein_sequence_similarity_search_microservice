import uuid
import pytest
from src.service.main import app
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# /health
def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_contains_all_version_fields(client):
    response = client.get("/health")

    body = response.json()

    assert "status" in body
    assert "service_version" in body
    assert "model_version" in body
    assert "corpus_version" in body
    assert "index_version" in body


def test_health_version_fields_are_non_empty(client):
    response = client.get("/health")

    body = response.json()

    assert body["status"] is not None
    assert body["service_version"] is not None
    assert body["model_version"] is not None
    assert body["corpus_version"] is not None
    assert body["index_version"]


# /ready
def test_ready_returns_200(client):
    response = client.get("/ready")
    assert response.status_code == 200


def test_ready_field_is_boolean_true(client):
    response = client.get("/ready")

    body = response.json()

    assert body["ready"] is True
    assert body["model_loaded"] is True
    assert body["index_loaded"] is True


# /embed
def test_embed_single_sequence_returns_correct_shape(client):
    response = client.post(
        "/embed", json={"sequences": ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"]}
    )
    output = response.json()["embeddings"]

    assert len(output) == 1
    assert len(output[0]) == 1280


def test_embed_two_sequences_returns_correct_shape(client):
    response = client.post(
        "/embed",
        json={
            "sequences": [
                "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP",
                "MKTAYIAKQRQISFVKSHFSRQHTQGYFP",
            ]
        },
    )
    output = response.json()["embeddings"]
    assert len(output) == 2
    assert len(output[0]) == 1280
    assert len(output[1]) == 1280


def test_embed_invalid_sequence_returns_invalid_sequence_code(client):
    response = client.post(
        "/embed", json={"sequences": ["MBCDKTAYIAKQRQISFVKSHFSRQHTQGYFP"]}
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_SEQUENCE"


def test_embed_sequence_over_limit_returns_sequence_too_long(client):
    response = client.post("/embed", json={"sequences": ["A" * 2000]})
    assert response.status_code == 422
    assert response.json()["error_code"] == "SEQUENCE_TOO_LONG"


def test_embed_empty_batch_returns_422(client):
    response = client.post("/embed", json={"sequences": []})
    assert response.status_code == 422


def test_embed_batch_over_limit_returns_batch_too_large(client):
    response = client.post("/embed", json={"sequences": ["AAA"] * 33})

    assert response.status_code == 422
    assert response.json()["error_code"] == "BATCH_TOO_LARGE"


def test_embed_response_contains_request_id(client):
    response = client.post(
        "/embed", json={"sequences": ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"]}
    )
    body = response.json()
    assert body["request_id"] is not None
    assert len(body["request_id"]) > 0


def test_embed_same_sequence_twice_returns_identical_embeddings(client):
    response = client.post(
        "/embed",
        json={
            "sequences": [
                "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP",
                "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP",
            ]
        },
    )
    output = response.json()["embeddings"]
    assert output[0] == output[1]


def test_success_response_request_id_is_valid_uuid(client):
    response = client.post(
        "/embed", json={"sequences": ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"]}
    )

    request_id = response.json()["request_id"]

    assert uuid.UUID(request_id)


def test_error_response_request_id_is_valid_uuid(client):
    response = client.post("/embed", json={"sequences": ["AAA"] * 33})

    request_id = response.json()["request_id"]

    assert uuid.UUID(request_id)


def test_request_ids_are_unique(client):
    request_ids = []

    for _ in range(10):
        response = client.post(
            "/embed", json={"sequences": ["MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"]}
        )

        request_ids.append(response.json()["request_id"])

    assert len(set(request_ids)) == 10


def test_embed_embedding_failed_returns_500(client):
    with patch(
        "src.service.routes.asyncio.to_thread", side_effect=RuntimeError("GPU OOM")
    ):
        response = client.post("/embed", json={"sequences": ["ACDEFGH"]})
    assert response.status_code == 500
    assert response.json()["error_code"] == "EMBEDDING_FAILED"


def test_embed_payload_too_large_returns_413(client):
    # max_payload_size from config — construct batch that exceeds it
    sequences = ["A" * 2000] * 30  # 60,000 chars > 50,000 default
    response = client.post("/embed", json={"sequences": sequences})
    assert response.status_code == 413
    assert response.json()["error_code"] == "PAYLOAD_TOO_LARGE"


# /search
def test_search_valid_sequence_returns_top_k_results(client):
    response = client.post(
        "/search", json={"sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"}
    )
    assert response.status_code == 200
    assert len(response.json()["results"]) == 10


def test_search_result_contains_required_fields(client):
    response = client.post(
        "/search", json={"sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"}
    )
    body = response.json()

    assert body["query_length"] is not None
    assert body["results"] is not None
    assert body["model_version"] is not None
    assert body["index_version"]
    assert body["request_id"] is not None

    result = body["results"][0]
    assert "rank" in result
    assert "accession" in result
    assert "score" in result
    assert "organism" in result
    assert "keywords" in result
    assert "go_terms" in result


def test_search_results_ordered_by_score(client):
    response = client.post(
        "/search", json={"sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP"}
    )
    results = response.json()["results"]
    assert results[0]["score"] >= results[1]["score"]


def test_search_invalid_sequence_returns_invalid_sequence_code(client):
    response = client.post(
        "/search", json={"sequence": "MBCDKTAYIAKQRQISFVKSHFSRQHTQGYFP"}
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_SEQUENCE"


def test_search_top_k_above_maximum_returns_422(client):
    response = client.post(
        "/search",
        json={"sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP", "top_k": 51},
    )
    assert response.status_code == 422


def test_search_search_failed_returns_500(client):
    with patch(
        "src.service.routes.asyncio.to_thread", side_effect=RuntimeError("FAISS error")
    ):
        response = client.post("/search", json={"sequence": "ACDEFGH"})
    assert response.status_code == 500
    assert response.json()["error_code"] == "SEARCH_FAILED"


def test_search_index_not_ready_returns_503(client):
    with patch.object(client.app.state, "index_loaded", False):
        response = client.post("/search", json={"sequence": "ACDEFGH"})
    assert response.status_code == 503
    assert response.json()["error_code"] == "INDEX_NOT_READY"


# /metrics
def test_metrics_returns_200(client):
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_returns_prometheus_content_type(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_total" in response.text
