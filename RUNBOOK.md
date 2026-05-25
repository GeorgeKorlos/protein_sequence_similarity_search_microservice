# RUNBOOK

## Local Development Setup

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (optional; CPU mode is supported)
- ~24 GB GPU memory for embedding inference (if using GPU)

### 1. Create Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
model_tag=facebook/esm2_t33_650M_UR50D
device=cuda
index_path=data/indexes/ivf
corpus_path=data/swissprot_clean.csv
ids_path=data/swissprot_ids.txt
max_batch_size=32
max_payload_size=50000
max_top_k=50
EOF
```

For CPU-only inference, set `device=cpu` (see **Operational Constraints** for latency implications).

### 4. Verify Setup

```bash
python -c "from src.core.embedder import ESM2Embedder; print('Setup OK')"
```

---

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_searcher.py -v
```

### Run with Coverage

```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

### Test Categories

- **Unit tests**: `test_embedder.py`, `test_validator.py`, `test_corpus_store.py`, `test_index_manager.py`
- **Integration tests**: `test_searcher.py`, `test_api_smoke.py`
- **Observability tests**: `test_metrics.py`, `test_schemas.py`

---

## Building the Docker Image

### 1. Build Locally

```bash
docker build -t protein-search:latest .
```

This Dockerfile:
- Installs `requirements-prod.txt` (optimized dependencies)
- Pre-downloads and caches the ESM-2 650M model weights at build time
- Copies the corpus metadata and embedding config
- **Does NOT** include the FAISS index — it is mounted at runtime

Build time: ~15–20 minutes (primarily model download + pip install).

### 2. Verify Image

```bash
docker images | grep protein-search
docker inspect protein-search:latest | grep -A 2 '"Env"'
```

### 3. Push to Registry (if using Cloud Run)

```bash
docker tag protein-search:latest gcr.io/PROJECT_ID/protein-search:latest
docker push gcr.io/PROJECT_ID/protein-search:latest
```

---

## Bringing Up the Full Stack

### Prerequisites

- Index files must be present at `data/indexes/ivf/` (see **Loading a Custom Index** below)
- `.env` file configured (see **Local Development Setup**)

### 1. Start Services with Docker Compose

```bash
docker-compose -f infra/docker-compose.yaml up
```

This brings up:
- **API service** (FastAPI on port 8000)
- **Prometheus** (metrics scraper on port 9090)

Index is mounted read-only from `data/indexes/ivf/` into the container.

### 2. Verify Services Are Running

```bash
docker-compose -f infra/docker-compose.yaml ps
```

### 3. View Logs

```bash
# API service logs
docker-compose -f infra/docker-compose.yaml logs -f api

# Prometheus logs
docker-compose -f infra/docker-compose.yaml logs -f prometheus
```

### 4. Stop the Stack

```bash
docker-compose -f infra/docker-compose.yaml down
```

---

## Verifying All Endpoints

### 1. Health Check

```bash
curl -s http://localhost:8000/health | jq .
```

Expected response:
```json
{
  "status": "ok",
  "service_version": "0.1.0",
  "model_version": "facebook/esm2_t33_650M_UR50D",
  "corpus_version": "2026_01",
  "index_version": {"index_type": "IndexIVFFlat", "params": {"nlist": 100, "nprobe": 10}}
}
```

### 2. Readiness Check

```bash
curl -s http://localhost:8000/ready | jq .
```

Expected response (when model and index are loaded):
```json
{
  "ready": true,
  "model_loaded": true,
  "index_loaded": true
}
```

Returns 503 Service Unavailable if model or index failed to load.

### 3. Embed a Sequence

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"sequences": ["MKVL", "ACDEFGHIKLMNPQRSTVWY"]}' | jq .
```

Expected response:
```json
{
  "embeddings": [[...], [...]], // 1280-dim vectors
  "model_version": "facebook/esm2_t33_650M_UR50D",
  "request_id": "uuid-string"
}
```

### 4. Search for Similar Sequences

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKVL", "top_k": 5}' | jq '.results[] | {rank, accession, score, organism}'
```

Expected response:
```json
[
  {"rank": 1, "accession": "P12345", "score": 0.98, "organism": "Homo sapiens"},
  {"rank": 2, "accession": "P23456", "score": 0.95, "organism": "Mus musculus"},
  ...
]
```

### 5. Prometheus Metrics

```bash
curl -s http://localhost:8000/metrics | head -30
```

Metrics include:
- `model_inference_seconds` — embedding latency
- `faiss_search_seconds` — search latency
- `embed_batch_size` — batch size distribution
- `errors_total` — error counts by type

### 6. OpenAPI Documentation

Navigate to [http://localhost:8000/docs](http://localhost:8000/docs) in a browser for interactive API explorer.

---

## Loading a Custom Index

### Option 1: Build Index from Embeddings (Recommended)

If you have embedding files (`.npy` + `.txt` IDs):

```bash
# Build IVFFlat index with nprobe=10 (default production config)
python scripts/build_faiss.py \
  --embeddings data/swissprot_embeddings.npy \
  --ids data/swissprot_ids.txt \
  --corpus data/swissprot_clean.csv \
  --output-dir data/indexes/ivf \
  --index-type IndexIVFFlat \
  --nlist 100 \
  --nprobe 10
```

This produces:
- `data/indexes/ivf/index.faiss` — FAISS index (binary)
- `data/indexes/ivf/index_meta.json` — metadata (index type, params, build date)

### Option 2: Use Existing Index

Copy the index directory into `data/indexes/` and configure the `.env`:

```bash
# .env
index_path=data/indexes/ivf  # or data/indexes/flat, data/indexes/hnsw, etc.
```

### Option 3: Build Alternative Index Types

```bash
# Flat (exact search, ground truth)
python scripts/build_faiss.py \
  --output-dir data/indexes/flat \
  --index-type IndexFlatIP

# HNSW (faster, approximate)
python scripts/build_faiss.py \
  --output-dir data/indexes/hnsw \
  --index-type IndexHNSWFlat \
  --m 32 \
  --ef-construction 200
```

Then update `.env` and restart the service.

### Verify Index

```bash
# Check index metadata
cat data/indexes/ivf/index_meta.json | jq .

# Check index file size
ls -lh data/indexes/ivf/index.faiss
```

---

## Known Operational Constraints

### 1. Cloud Run Cold Start

**Issue**: First request after service startup takes 60–90 seconds.

**Cause**: ESM-2 model weights are baked into the Docker image at build time via `AutoModel.from_pretrained()`. Cold start includes GCS index download (~2.7 GB, adds 60–90s) followed by model deserialization and corpus load into CPU memory.

**Mitigation**:
- Model is pre-downloaded and cached at build time (see Dockerfile), so no network download occurs at runtime
- The GCS index download dominates cold start time (~60–90s); model load adds ~40s on CPU
- Set Cloud Run minimum instances to 1 (costs ~$7/month) to keep the service warm and avoid cold starts entirely

### 2. Memory Limit (16 GiB on Cloud Run)

**Issue**: Service can run out of memory if index + model + corpus are too large.

**Current footprint**:
- ESM-2 650M model: ~1.3 GB (float16, CPU; loaded via low_cpu_mem_usage=True)
- Corpus (547K sequences metadata): ~50 MB
- FAISS IVFFlat index: ~2.7 GB (CPU RAM, loaded on startup)
- Request buffers: ~100–500 MB

**Total**: ~8.2 GB (measured peak at startup — exceeded 8Gi limit on two revisions; 16Gi required)

**Mitigation**:
- **Avoid large batch `/embed` requests** — max 256 sequences, but CPU-only mode will spike memory
- **Do not use Flat index in production** — it requires 2.7 GB per search query
- **Index selection is critical** — use IVFFlat with nprobe=10 (2.7 GB) or HNSW (2.8 GB, faster)
- **Monitor memory**: Prometheus metric `process_resident_memory_bytes` available at `/metrics`

### 3. CPU-Only Inference Latency

**Issue**: Setting `device=cpu` causes severe latency degradation.

**Latency impact** (measured, single sequence):
- **GPU (CUDA)**: ~10–15 ms per sequence (benchmarked on RTX 3090)
- **CPU**: ~23s p50, ~24.8s p95 (measured on Cloud Run, europe-west1, 4 vCPU, warm instance)
- **Speedup**: CPU is **~1500–2500x slower** than GPU

**Throughput impact** (per `/embed` call, batch of 32 sequences):
- **GPU**: ~80 sequences/sec
- **CPU**: <0.1 sequences/sec (measured)

**When to use CPU mode**:
- Local development (no GPU available)
- Debugging (offload model onto laptop for rapid iteration)
- **NOT recommended** for production — violates SLA if interactive latency < 1 second

**Production recommendation**: Always use GPU. If GPU is unavailable, use a smaller model (ESM2-150M) or reduce batch size to mitigate latency.

### 4. Index Load Time

FAISS index is loaded synchronously into memory at service startup. Load time is <5 seconds for IVFFlat/HNSW (2.7–2.8 GB). Service readiness endpoint (`/ready`) returns 503 until index is fully loaded.

### 5. Search Latency by Index Type

From benchmark (500 query evaluation):

| Index Type | Configuration | Median Latency | QPS |
|---|---|---|---|
| Flat | — | 43 ms | 23 |
| IVFFlat | nprobe=10 | 22 ms | 45 |
| HNSW | efSearch=64 | 0.6 ms | 1677 |

**Trade-offs**:
- **Flat**: Perfect recall (1.0), slow for large corpora
- **IVFFlat**: 99% recall, ~20ms latency (suitable for interactive APIs)
- **HNSW**: Fastest (~0.6ms), but requires tuning `efSearch` for accuracy

---

## Cloud Run Operations

**Live service URL**: https://protein-search-699950260063.europe-west1.run.app

### Deploy a new image version
```bash
docker build -t protein-search:latest .
docker tag protein-search:latest \
  europe-west1-docker.pkg.dev/protein-search-497311/protein-search/protein-search:latest
docker push \
  europe-west1-docker.pkg.dev/protein-search-497311/protein-search/protein-search:latest
gcloud run deploy protein-search \
  --image europe-west1-docker.pkg.dev/protein-search-497311/protein-search/protein-search:latest \
  --region europe-west1
```

### Check service status
```bash
gcloud run services describe protein-search --region europe-west1
```

### Check Cloud Run logs
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=protein-search" \
  --limit 50 \
  --format "value(textPayload)"
```

### Roll back to previous revision
```bash
# List revisions
gcloud run revisions list --service protein-search --region europe-west1

# Route traffic to a specific revision
gcloud run services update-traffic protein-search \
  --region europe-west1 \
  --to-revisions REVISION_NAME=100
```

### Tear down (cost cleanup)
```bash
# Delete Cloud Run service
gcloud run services delete protein-search --region europe-west1

# Delete index from GCS
gsutil rm -r gs://protein-search-497311-index

# Delete image from Artifact Registry
gcloud artifacts docker images delete \
  europe-west1-docker.pkg.dev/protein-search-497311/protein-search/protein-search --delete-tags
```

## Troubleshooting

### Service Won't Start

```bash
# Check service logs
docker-compose -f infra/docker-compose.yaml logs api --tail 50

# Common issues:
# - Index not found: Ensure data/indexes/ivf/index.faiss exists
# - Model download timeout: Check internet connectivity
# - OOM: Check Docker memory allocation (set to 10+ GB)
```

### Model Failed to Load

```bash
# Test model load directly
python -c "from src.core.embedder import ESM2Embedder; \
  e = ESM2Embedder('facebook/esm2_t33_650M_UR50D', device='cuda'); \
  print(e.embed(['MKVL']))"
```

### Index Not Loaded

```bash
# Check index metadata
python -c "from src.core.index_manager import IndexManager; \
  m = IndexManager('', dim=0, params=None); \
  m.load('data/indexes/ivf'); \
  print(f'Index ntotal: {m.index.ntotal}')"
```

### Slow Search Results

```bash
# Check if using CPU instead of GPU
curl http://localhost:8000/health | grep model_version

# Then test embedding latency:
time curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"sequences": ["MKVL"]}' > /dev/null

# If > 200ms, device is likely CPU. Restart with device=cuda.
```

---

## Summary

| Task | Command |
|---|---|
| Install dependencies | `pip install -r requirements.txt` |
| Run tests | `pytest tests/ -v` |
| Build Docker image | `docker build -t protein-search:latest .` |
| Start services | `docker-compose -f infra/docker-compose.yaml up` |
| Health check | `curl http://localhost:8000/health` |
| Search | `curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"sequence": "MKVL", "top_k": 10}'` |
| Build index | `python scripts/build_faiss.py --index-type IndexIVFFlat` |
| View metrics | `http://localhost:9090` (Prometheus) |
