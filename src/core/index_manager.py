import json
import faiss
import hashlib
import datetime
import numpy as np
from pathlib import Path


class IndexManager:
    """Manages a FAISS index instance; index_type selects the FAISS index variant (e.g., flat, hnsw, ivf)."""

    def __init__(self, index_type, dim, params: dict | None) -> None:
        self.index_type = index_type
        self.dim = dim
        self.params = params or {}

        self.corpus_version = "unknown"
        self.model_version = "unknown"

        self.index = None
        self.is_trained = False
        self.built_at = None
        self.checksum = None

    def build(self, embeddings: np.ndarray) -> None:
        if embeddings.dtype != "float32":
            embeddings = embeddings.astype("float32")

        if self.index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(self.dim)

            self.index.add(embeddings)  # type: ignore[arg-type]
            self.is_trained = True
            self.built_at = datetime.datetime.utcnow().isoformat()

        elif self.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = self.params.get("nlist", 100)

            self.index = faiss.IndexIVFFlat(
                quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.nprobe = self.params.get("nprobe", 10)
            self.index.train(embeddings)  # type: ignore[arg-type]
            self.is_trained = True
            self.index.add(embeddings)  # type: ignore[arg-type]
            self.built_at = datetime.datetime.utcnow().isoformat()

        elif self.index_type == "IndexHNSWFlat":
            M = self.params.get("M", 32)
            self.index = faiss.IndexHNSWFlat(self.dim, M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = self.params.get("efConstruction", 200)
            self.index.add(embeddings)  # type: ignore[arg-type]
            self.is_trained = True
            self.built_at = datetime.datetime.utcnow().isoformat()

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise ValueError("Index not built")

        index_file = path / "index.faiss"
        meta_file = path / "index_meta.json"

        faiss.write_index(self.index, str(index_file))

        with open(index_file, "rb") as f:
            index_bytes = f.read()

        index_hash = hashlib.sha256(index_bytes).hexdigest()
        self.checksum = index_hash

        meta = {
            "index_type": self.index_type,
            "dim": self.dim,
            "params": self.params,
            "is_trained": self.is_trained,
            "built_at": self.built_at,
            "checksum": index_hash,
            "corpus_version": self.corpus_version,
            "model_version": self.model_version,
        }

        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str | Path):
        path = Path(path)

        index_file = path / "index.faiss"
        meta_file = path / "index_meta.json"

        with open(meta_file, "r") as f:
            meta = json.load(f)

        with open(index_file, "rb") as f:
            index_bytes = f.read()

        index_hash = hashlib.sha256(index_bytes).hexdigest()

        if index_hash != meta["checksum"]:
            raise ValueError("Index corrupted")

        self.index = faiss.read_index(str(index_file))
        self.index_type = meta["index_type"]
        self.dim = meta["dim"]
        self.params = meta["params"]
        self.is_trained = meta["is_trained"]
        self.built_at = meta["built_at"]
        self.checksum = meta["checksum"]
        self.corpus_version = meta.get("corpus_version", "unknown")
        self.model_version = meta.get("model_version", "unknown")

    @property
    def index_version(self):
        return {
            "checksum": self.checksum,
            "build_timestamp": self.built_at,
            "index_type": self.index_type,
            "params": self.params,
            "n_vectors": self.index.ntotal if self.index is not None else None,
        }
