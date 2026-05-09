import os
import time
import json
import argparse
import tempfile
import numpy as np
from pathlib import Path
from src.core.index_manager import IndexManager


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/swissprot_embeddings.npy"),
        help="Path to the embeddings NPY file (default: data/swissprot_embeddings.npy)",
    )

    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("data/swissprot_ids.txt"),
        help="Path to the id file (default: data/swissprot_ids.txt)",
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/swissprot_clean.csv"),
        help="Path to the input CSV file (default: data/swissprot_clean.csv)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/"),
        help="Directory to the benchmark results (default: reports/)",
    )

    parser.add_argument(
        "--n-queries",
        type=int,
        default=1,
        help="How many vectors to hold out as queries (default: 1)",
    )

    parser.add_argument(
        "--k", type=int, default=5, help="top-k for search (default: 5)"
    )

    return parser.parse_args()


def compute_recall(retrieved, ground_truth, index_ids):
    recalls = []

    for i, row in enumerate(retrieved):
        predicted = {index_ids[int(pos)] for pos in row}
        correct = predicted.intersection(ground_truth[i])

        recalls.append(len(correct) / len(ground_truth[i]))

    return float(np.mean(recalls))


def measure_qps(index, query_embeddings, k, n_queries):
    qps_runs = []

    for _ in range(3):
        start_time = time.time()

        index.search(query_embeddings, k)

        wall_time = time.time() - start_time

        qps_runs.append(n_queries / wall_time)

    return (
        float(np.mean(qps_runs)),
        float(np.median(qps_runs)),
        float(np.std(qps_runs)),
    )


def measure_index_size(index_manager):
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "index"

        index_manager.save(index_path)

        size_mb = os.path.getsize(index_path / "index.faiss") / (1024 * 1024)

    return float(size_mb)


def main():
    args = parse_args()

    np.random.seed(42)

    embeddings = np.load(args.embeddings, mmap_mode="r")

    with open(args.ids, "r") as f:
        ids = np.array([line.strip() for line in f if line.strip()])

    assert len(ids) == len(embeddings)

    query_set = np.random.choice(
        len(embeddings),
        args.n_queries,
        replace=False,
    )

    index_set = np.setdiff1d(
        np.arange(len(embeddings)),
        query_set,
    )

    query_embeddings = embeddings[query_set]
    index_embeddings = embeddings[index_set]

    index_ids = ids[index_set]

    results = []

    # FlatIP — ground truth baseline

    flat_manager = IndexManager(
        index_type="IndexFlatIP",
        dim=1280,
        params={},
    )

    start_time = time.time()

    flat_manager.build(index_embeddings)

    flat_build_time = time.time() - start_time

    _, gt_indices = flat_manager.index.search(  # type: ignore
        query_embeddings,
        args.k,
    )

    ground_truth = {}

    for i, row in enumerate(gt_indices):
        ground_truth[i] = {index_ids[int(pos)] for pos in row}

    (
        flat_qps_mean,
        flat_qps_median,
        flat_qps_std,
    ) = measure_qps(
        flat_manager.index,  # type: ignore
        query_embeddings,
        args.k,
        args.n_queries,
    )

    flat_size_mb = measure_index_size(flat_manager)

    results.append(
        {
            "index_type": "IndexFlatIP",
            "params": {},
            "recall_at_k": 1.0,
            "qps_mean": flat_qps_mean,
            "qps_median": flat_qps_median,
            "qps_std": flat_qps_std,
            "build_time_s": flat_build_time,
            "index_size_mb": flat_size_mb,
        }
    )

    # IVFFlat

    for nprobe in [1, 5, 10, 20, 50]:

        ivf_manager = IndexManager(
            "IndexIVFFlat",
            1280,
            {
                "nlist": 100,
                "nprobe": nprobe,
            },
        )

        start_time = time.time()

        ivf_manager.build(index_embeddings)

        build_time = time.time() - start_time

        _, indices = ivf_manager.index.search(  # type: ignore
            query_embeddings,
            args.k,
        )

        recall = compute_recall(
            indices,
            ground_truth,
            index_ids,
        )

        (
            qps_mean,
            qps_median,
            qps_std,
        ) = measure_qps(
            ivf_manager.index,  # type: ignore
            query_embeddings,
            args.k,
            args.n_queries,
        )

        size_mb = measure_index_size(ivf_manager)

        results.append(
            {
                "index_type": "IndexIVFFlat",
                "params": {
                    "nlist": 100,
                    "nprobe": nprobe,
                },
                "recall_at_k": recall,
                "qps_mean": qps_mean,
                "qps_median": qps_median,
                "qps_std": qps_std,
                "build_time_s": build_time,
                "index_size_mb": size_mb,
            }
        )

    # HNSW

    for ef in [16, 32, 64, 128]:

        hnsw_manager = IndexManager(
            "IndexHNSWFlat",
            1280,
            {
                "M": 32,
                "efConstruction": 200,
            },
        )

        start_time = time.time()

        hnsw_manager.build(index_embeddings)

        build_time = time.time() - start_time

        hnsw_manager.index.hnsw.efSearch = ef  # type: ignore

        _, indices = hnsw_manager.index.search(  # type: ignore
            query_embeddings,
            args.k,
        )

        recall = compute_recall(
            indices,
            ground_truth,
            index_ids,
        )

        (
            qps_mean,
            qps_median,
            qps_std,
        ) = measure_qps(
            hnsw_manager.index,  # type: ignore
            query_embeddings,
            args.k,
            args.n_queries,
        )

        size_mb = measure_index_size(hnsw_manager)

        results.append(
            {
                "index_type": "IndexHNSWFlat",
                "params": {
                    "M": 32,
                    "efConstruction": 200,
                    "efSearch": ef,
                },
                "recall_at_k": recall,
                "qps_mean": qps_mean,
                "qps_median": qps_median,
                "qps_std": qps_std,
                "build_time_s": build_time,
                "index_size_mb": size_mb,
            }
        )

    # Save results

    os.makedirs(args.output_dir, exist_ok=True)

    json_path = args.output_dir / "benchmark_results.json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    markdown_lines = []
    markdown_lines.append("# Benchmark Results")
    markdown_lines.append("")

    markdown_lines.append(
        "| Index Type | Params | Recall@k | QPS Mean | QPS Median | QPS Std | Build Time (s) | Index Size (MB) |"
    )

    markdown_lines.append("|---|---|---|---|---|---|---|---|")

    for row in results:

        markdown_lines.append(
            f"| "
            f"{row['index_type']} | "
            f"`{row['params']}` | "
            f"{row['recall_at_k']:.4f} | "
            f"{row['qps_mean']:.2f} | "
            f"{row['qps_median']:.2f} | "
            f"{row['qps_std']:.2f} | "
            f"{row['build_time_s']:.2f} | "
            f"{row['index_size_mb']:.2f} |"
        )

    markdown_content = "\n".join(markdown_lines)
    markdown_path = args.output_dir / "benchmark_results.md"

    with open(markdown_path, "w") as f:
        f.write(markdown_content)

    print()
    print("Benchmark complete")
    print(f"Total embeddings: {embeddings.shape[0]}")
    print(f"Query set: {len(query_set)}")
    print(f"Index set: {len(index_set)}")
    print(f"k: {args.k}")
    print()

    for row in results:
        print(row)

    print()
    print(f"JSON results: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
