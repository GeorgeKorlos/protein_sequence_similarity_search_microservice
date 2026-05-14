import numpy as np
from src.core.exceptions import IndexNotReadyException


class Searcher:

    def __init__(self, corpus_store, embedder, index_manager) -> None:
        self.corpus_store = corpus_store
        self.embedder = embedder
        self.index_manager = index_manager

        if self.index_manager.is_trained is False:
            raise IndexNotReadyException

    def search(self, sequence: str, k: int = 10):
        query_vector = self.embedder.embed([sequence])
        query_vector = query_vector.squeeze()
        assert np.isclose(np.linalg.norm(query_vector), 1.0, atol=1e-5)
        query_vector = np.expand_dims(query_vector, axis=0)
        distances, indices = self.index_manager.index.search(query_vector, k)
        distances = distances.squeeze()
        indices = indices.squeeze()
        results = []
        for rank, (idx, score) in enumerate(zip(indices, distances), start=1):
            if idx == -1:
                continue
            metadata = self.corpus_store.get_metadata(idx)
            results.append(
                {
                    "rank": rank,
                    "accession": metadata["id"],
                    "score": float(score),
                    "organism": metadata["organism"],
                    "keywords": metadata["keywords"],
                    "go_terms": metadata["go_terms"],
                }
            )
        return results
