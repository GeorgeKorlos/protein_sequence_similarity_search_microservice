import hashlib
import pandas as pd
from pathlib import Path


class CorpusStore:
    def __init__(
        self,
        csv_path: str | Path,
        ids_path: str | Path | None = None,
        nrows: int | None = None,
    ):
        with open(csv_path, "rb") as f:
            self._corpus_version = hashlib.sha256(f.read()).hexdigest()

        self.df = pd.read_csv(csv_path, nrows=nrows)

        if ids_path is not None:
            with open(ids_path) as f:
                ordered_ids = [line.strip() for line in f if line.strip()]
            self.df = self.df.set_index("id").loc[ordered_ids].reset_index()

    @property
    def corpus_version(self) -> str:
        return self._corpus_version

    def __len__(self) -> int:
        return len(self.df)

    def get_sequence(self, idx: int) -> str:
        sequence = self.df["sequence"].iloc[idx]
        return sequence

    # id, organism, keywords, go_terms
    def get_metadata(self, idx: int) -> dict:
        metadata = self.df.iloc[idx][
            ["id", "organism", "keywords", "go_terms"]
        ].to_dict()
        return metadata

    def get_all_sequences(self) -> list[str]:
        sequence_list = self.df["sequence"].to_list()
        return sequence_list

    def get_all_ids(self) -> list[str]:
        id_list = self.df["id"].to_list()
        return id_list
