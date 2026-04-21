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
    MODEL_TAG = 'facebook/esm2_t30_150M_UR50D'
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Device selected %s", self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(ESM2Embedder.MODEL_TAG)
        self.model = AutoModel.from_pretrained(ESM2Embedder.MODEL_TAG).to(self.device)
        self.model.eval()
        self._model_version = self._compute_model_version()
        

        if self.debug:
            ok = self.verify_determinism('ACDE')
            logger.info(f"Determinism check passed: {ok}")


    def _compute_model_version(self) -> str:
        config_dict = self.model.config.to_dict()
        config_json = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        return f"{ESM2Embedder.MODEL_TAG}_{config_hash[:8]}"
    

    @property
    def model_version(self) -> str:
        return self._model_version
    

    def embed(self, sequences: list[str]) -> np.ndarray:
        tokens = self.tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device) 

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

        pooled = self._mean_pool(last_hidden_state, attention_mask)
        normalized = pooled / pooled.norm(p=2, dim=1, keepdim=True)

        embeddings = (normalized.cpu().numpy().astype(np.float32))
        return embeddings


    def _mean_pool(self, hidden_states, attention_mask):
        hs = hidden_states[:, 1: -1, :]
        mask = attention_mask[:, 1: -1]

        mask = mask.unsqueeze(-1)
        summed = (hs * mask).sum(dim=1)

        counts = mask.sum(dim=1)
        counts = counts.clamp(min=1)

        pooled = summed / counts

        zero_mask = (pooled.abs().sum(dim=1) == 0)
        if zero_mask.any():
            logger.warning(f"{zero_mask.sum().item()} sequences produced zero embeddings")
        
        return pooled


    def verify_determinism(self, sequence: str) -> bool:
        result_a = self.embed([sequence])
        result_b = self.embed([sequence])
        return np.allclose(result_a, result_b, atol=1e-6)
