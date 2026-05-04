import pytest
import numpy as np
from src.core.index_manager import IndexManager


@pytest.fixture
def embeddings():
    vecs = np.random.rand(1000, 1280).astype("float32")
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norm


def test_build_flat_ip(embeddings):
    im_flat = IndexManager("IndexFlatIP", 1280, {})
    im_flat.build(embeddings)
    assert im_flat.index.ntotal == 1000  # type: ignore


def test_build_hnsw(embeddings):
    im_hnsw = IndexManager("IndexHNSWFlat", 1280, {"M": 32})
    im_hnsw.build(embeddings)
    assert im_hnsw.index.ntotal == 1000  # type: ignore


def test_build_ivf(embeddings):
    im_ivf = IndexManager("IndexIVFFlat", 1280, {"nlist": 100, "nprobe": 10})
    im_ivf.build(embeddings)
    assert im_ivf.index.ntotal == 1000  # type: ignore


@pytest.mark.skip
def test_ivf_is_trained():
    pass


def test_save_load_flat_ip(embeddings, tmp_path):
    im_flat = IndexManager("IndexFlatIP", 1280, {})
    im_flat.build(embeddings)

    flat_path = tmp_path
    im_flat.save(flat_path)

    im_flat2 = IndexManager("IndexFlatIP", 1280, {})
    im_flat2.load(flat_path)
    assert im_flat2.index.ntotal == 1000  # type: ignore[attr-defined]
    assert im_flat2.index_version["n_vectors"] == 1000


def test_save_load_hnsw(embeddings, tmp_path):
    im_hnsw = IndexManager("IndexHNSWFlat", 1280, {"M": 32})
    im_hnsw.build(embeddings)

    hnsw_path = tmp_path
    im_hnsw.save(hnsw_path)

    im_hnsw2 = IndexManager("IndexHNSWFlat", 1280, {"M": 32})
    im_hnsw2.load(hnsw_path)

    assert im_hnsw2.index.ntotal == 1000  # type: ignore[attr-defined]
    assert im_hnsw2.index_version["n_vectors"] == 1000


def test_save_load_ivf(embeddings, tmp_path):
    im_ivf = IndexManager("IndexIVFFlat", 1280, {"nlist": 100, "nprobe": 10})
    im_ivf.build(embeddings)

    ivf_path = tmp_path
    im_ivf.save(ivf_path)

    im_ivf2 = IndexManager("IndexIVFFlat", 1280, {"nlist": 100, "nprobe": 10})
    im_ivf2.load(ivf_path)

    assert im_ivf2.index.ntotal == 1000  # type: ignore[attr-defined]
    assert im_ivf2.index_version["n_vectors"] == 1000


def test_index_version_keys(embeddings):
    im_flat = IndexManager("IndexFlatIP", 1280, {})
    im_flat.build(embeddings)
    index_version = im_flat.index_version
    assert {
        "checksum",
        "build_timestamp",
        "index_type",
        "params",
        "n_vectors",
    } <= index_version.keys()
    assert index_version["build_timestamp"] is not None
    assert index_version["index_type"] is not None
    assert index_version["params"] is not None
    assert index_version["n_vectors"] is not None


def test_checksum_consistency(embeddings, tmp_path):
    im_flat = IndexManager("IndexFlatIP", 1280, {})
    im_flat.build(embeddings)

    im_path = tmp_path
    im_flat.save(im_path)

    im_flat2 = IndexManager("IndexFlatIP", 1280, {})
    im_flat2.load(im_path)

    assert im_flat2.checksum == im_flat.checksum
