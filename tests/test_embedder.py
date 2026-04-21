from src.core.embedder import ESM2Embedder
import numpy as np
import torch

embedder = ESM2Embedder()
sequence = "MKTAYIAKQRQISFVKSHFSRQ"
output = embedder.embed(sequences=[sequence])

def test_embedder_output_shape():
    assert output.shape == (1, 640)


def test_embedder_output_type():
    assert output.dtype == np.float32


def test_embedder_l2_norm():
    assert np.isclose(np.linalg.norm(output[0]), 1.0, atol=1e-5)


def test_embedder_verify_determinism():
    output_a = embedder.embed(sequences=[sequence])
    output_b = embedder.embed(sequences=[sequence])
    assert np.array_equal(output_a, output_b)