from src.core.embedder import ESM2Embedder, PhysiochemicalEmbedder, RandomEmbedder
import numpy as np

embedder = ESM2Embedder(compile_model=False)
phys_emb = PhysiochemicalEmbedder()
rand_emb_42 = RandomEmbedder(seed=42)

sequence = "MKTAYIAKQRQISFVKSHFSRQ"
output = embedder.embed(sequences=[sequence])


def test_embedder_output_shape():
    assert output.shape == (1, 1280)


def test_embedder_output_type():
    assert output.dtype == np.float32


def test_embedder_l2_norm():
    assert np.isclose(np.linalg.norm(output[0]), 1.0, atol=1e-4)


def test_embedder_verify_determinism():
    output_a = embedder.embed(sequences=[sequence])
    output_b = embedder.embed(sequences=[sequence])
    assert np.array_equal(output_a, output_b)


def test_embedding_dim():
    assert embedder.embedding_dim == 1280


def test_physiochemical_embedder_output_shape():
    out = phys_emb.embed(["ACDE", "MKTAY"])
    assert out.shape == (2, 4)


def test_physiochemical_embedder_l2_norm():
    out = phys_emb.embed(["ACDEFGHIK"])
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)


def test_physiochemical_embedder_unknown_residue():
    out = phys_emb.embed(["ACXDE"])
    assert out.shape == (1, 4)
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)


def test_random_embedder_output_shape():
    out = rand_emb_42.embed(["ACDE", "MKTAY"])
    assert out.shape == (2, 1280)


def test_random_embedder_l2_norm():
    out = rand_emb_42.embed(["ACDEFGHIK"])
    assert np.isclose(np.linalg.norm(out[0]), 1.0, atol=1e-5)


def test_random_embedder_reproducibility():
    out_a = RandomEmbedder(seed=42).embed(["ACDE"])
    out_b = RandomEmbedder(seed=42).embed(["ACDE"])
    assert np.array_equal(out_a, out_b)


def test_random_embedder_different_seeds_differ():
    out_a = RandomEmbedder(seed=42).embed(["ACDE"])
    out_b = RandomEmbedder(seed=99).embed(["ACDE"])
    assert not np.array_equal(out_a, out_b)
