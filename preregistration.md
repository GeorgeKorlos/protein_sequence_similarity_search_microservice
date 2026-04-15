# Preregistration

## Project Goal

The goal of this project is to build a protein sequence similarity search system using ESM-2 embeddings and benchmark FAISS index types for efficient retrieval.

## Corpus 

SwissProt reviewed subset (UniProt).
Rationale: curated, high-quality and widely accepted benchmark dataset with reliable functional annotations.

## Sequence Filter Criteria

- Remove sequences annotated as fragments
- Cap sequence length at 1024 amino acids

## Embedding Strategy

- Model: ESM-2 (esm2_t30_150M_ur50d)
- Representation: Mean pooling over token embeddings
- Post-processing: L2 normalization

## Similarity Metric

Cosine Similarity

## Index Types to Benchmark

- FAISS IndexFlat (exact search, ground truth)
- FAISS IndexIVFFlat
- FAISS IndexHNSWFlat

## Benchmark Success Criteria

Recall@10 ≥ 0.95 for HNSW compared to Flat index ground truth

If Recall@10 < 0.95, IVFFlat will be evaluated as the production index candidate

## P5 Handoff

Export embeddings as `.npy` files for downstream Drug–Target Interaction (DTI) model
