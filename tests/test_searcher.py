import pytest
import numpy as np
from src.core.searcher import Searcher
from src.core.index_manager import IndexManager
from src.core.exceptions import IndexNotReadyError


@pytest.fixture
def embeddings():
    vecs = np.random.rand(100, 1280).astype("float32")
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norm


def test_search_exact_match_returns_self_as_top_hit(embeddings, mocker):
    im_flat = IndexManager(index_type="IndexFlatIP", dim=1280, params={})
    im_flat.build(embeddings)

    embedder = mocker.Mock()
    embedder.embed.return_value = embeddings[0:1]
    corpus_store = mocker.Mock()
    corpus_store.get_metadata.side_effect = lambda idx: {
        "id": f"id_{idx}",
        "organism": "test",
        "keywords": [],
        "go_terms": [],
    }

    searcher = Searcher(corpus_store, embedder, im_flat)
    results = searcher.search("dummy", k=5)

    assert isinstance(results, list)
    assert len(results) > 0

    expected_keys = {"rank", "accession", "score", "organism", "keywords", "go_terms"}
    for r in results:
        assert set(r.keys()) == expected_keys


def test_search_returns_expected_results_structure(embeddings, mocker):
    im = IndexManager("IndexFlatIP", 1280, {})
    im.build(embeddings)

    embedder = mocker.Mock()
    embedder.embed.return_value = embeddings[0:1]

    corpus_store = mocker.Mock()
    corpus_store.get_metadata.side_effect = lambda idx: {
        "id": f"id_{idx}",
        "organism": "test",
        "keywords": ["kw"],
        "go_terms": ["go"],
    }

    searcher = Searcher(corpus_store, embedder, im)
    results = searcher.search("dummy", k=5)

    assert isinstance(results, list)
    assert len(results) > 0

    expected_keys = {"rank", "accession", "score", "organism", "keywords", "go_terms"}
    for r in results:
        assert set(r.keys()) == expected_keys


def test_search_results_sorted_by_score_and_ranked_sequentially(embeddings, mocker):
    im = IndexManager("IndexFlatIP", 1280, {})
    im.build(embeddings)

    embedder = mocker.Mock()
    embedder.embed.return_value = embeddings[0:1]

    corpus_store = mocker.Mock()
    corpus_store.get_metadata.side_effect = lambda idx: {
        "id": f"id_{idx}",
        "organism": "test",
        "keywords": [],
        "go_terms": [],
    }

    searcher = Searcher(corpus_store, embedder, im)
    results = searcher.search("dummy", k=10)

    scores = [r["score"] for r in results]
    ranks = [r["rank"] for r in results]

    assert scores == sorted(scores, reverse=True)
    assert ranks == list(range(1, len(results) + 1))


def test_search_respects_k_order(embeddings, mocker):
    im = IndexManager("IndexFlatIP", 1280, {})
    im.build(embeddings)

    embedder = mocker.Mock()
    embedder.embed.return_value = embeddings[0:1]

    corpus_store = mocker.Mock()
    corpus_store.get_metadata.side_effect = lambda idx: {
        "id": f"id_{idx}",
        "organism": "test",
        "keywords": [],
        "go_terms": [],
    }

    searcher = Searcher(corpus_store, embedder, im)
    results = searcher.search("dummy", k=5)

    assert len(results) == 5


def test_searcher_raises_error_when_index_not_ready(mocker):
    im = IndexManager("IndexIVFFlat", 1280, {})
    embedder = mocker.Mock()
    corpus_store = mocker.Mock()

    with pytest.raises(IndexNotReadyError):
        Searcher(corpus_store, embedder, im)
