import abc
import json
import numpy as np
import torch
import hashlib
import logging
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class BaseEmbedder(abc.ABC):
    @abc.abstractmethod
    def embed(self, sequences: list[str]) -> np.ndarray:
        raise NotImplementedError


class ESM2Embedder(BaseEmbedder):
    MODEL_TAG = "facebook/esm2_t33_650M_UR50D"

    def __init__(
        self,
        device: str | None = None,
        debug: bool = False,
        compile_model: bool = True,
    ):
        self.debug = debug

        if device is not None:
            if device not in ("cpu", "cuda"):
                raise ValueError(f"Invalid device: {device}")
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available")
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        logger.info("Device selected: %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            ESM2Embedder.MODEL_TAG, use_fast=True
        )

        model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            ESM2Embedder.MODEL_TAG, dtype=model_dtype
        ).to(self.device)
        self.model.eval()

        if compile_model and self.device.type == "cuda":
            self.model = torch.compile(self.model)
            logger.info("Model compiled with torch.compile")

        self._model_version = self._compute_model_version()

        if self.debug:
            ok = self.verify_determinism("ACDE")
            logger.info(f"Determinism check passed: {ok}")

    def _compute_model_version(self) -> str:
        config_dict = self.model.config.to_dict()
        config_json = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        return f"{ESM2Embedder.MODEL_TAG}_{config_hash[:8]}"

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            sequences, padding=True, truncation=True, return_tensors="pt"
        )
        return {k: v.pin_memory() for k, v in tokens.items()}

    def embed_tokenized(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        normalized = pooled / pooled.norm(p=2, dim=1, keepdim=True)
        return normalized.float().cpu().numpy()

    def embed(self, sequences: list[str]) -> np.ndarray:
        tokens = self.tokenize(sequences)
        input_ids = tokens["input_ids"].to(self.device, non_blocking=True)
        attention_mask = tokens["attention_mask"].to(self.device, non_blocking=True)
        return self.embed_tokenized(input_ids, attention_mask)

    def _mean_pool(self, hidden_states, attention_mask):
        hs = hidden_states[:, 1:-1, :]
        mask = attention_mask[:, 1:-1].unsqueeze(-1)
        summed = (hs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = summed / counts

        zero_mask = pooled.abs().sum(dim=1) == 0
        if zero_mask.any():
            logger.warning(
                f"{zero_mask.sum().item()} sequences produced zero embeddings"
            )

        return pooled

    def verify_determinism(self, sequence: str) -> bool:
        result_a = self.embed([sequence])
        result_b = self.embed([sequence])
        return np.allclose(result_a, result_b, atol=1e-6)
