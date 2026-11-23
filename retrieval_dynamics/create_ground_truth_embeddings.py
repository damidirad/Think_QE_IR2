
import faiss
import numpy as np
import pandas as pd
from pathlib import Path

base = Path("/scratch-shared/scur0915")

qrels_dir = base / "topics-and-qrels"

faiss_dir = next(
    (base / "pyserini_cache/indexes").glob(
        "faiss.msmarco-v1-passage.sbert.*"
    )
)

out_dir = base / "embedding_analysis"
out_dir.mkdir(exist_ok=True)

# Load ground-truth relevance judgments
frames = []

for dataset in ["dl19", "dl20"]:
    df = pd.read_csv(
        qrels_dir / f"qrels.{dataset}-passage.txt",
        sep=r"\s+",
        names=["query_id", "unused", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str}
    )

    df["dataset"] = dataset
    frames.append(df)

qrels = pd.concat(frames, ignore_index=True)

target_ids = set(qrels.doc_id)

print("Total judgments:", len(qrels))
print("Unique ground-truth documents:", len(target_ids))

# Find document positions in FAISS
positions = {}

with open(faiss_dir / "docid") as f:
    for pos, line in enumerate(f):
        docid = line.strip()

        if docid in target_ids:
            positions[docid] = pos

missing = target_ids - positions.keys()

print("Matched documents:", len(positions))
print("Missing documents:", len(missing))
print("Coverage:", len(positions) / len(target_ids))

# Load full FAISS index
print("Loading FAISS index...")

faiss.omp_set_num_threads(4)
index = faiss.read_index(str(faiss_dir / "index"))

# Extract selected embeddings
docids = sorted(positions)

embeddings = np.array([
    index.reconstruct(positions[docid])
    for docid in docids
], dtype=np.float32)

print("Embedding shape:", embeddings.shape)

# Save extracted subset
np.savez(
    out_dir / "ground_truth_embeddings.npz",
    docids=np.array(docids),
    embeddings=embeddings
)

qrels.to_csv(
    out_dir / "ground_truth_qrels.csv",
    index=False
)

print("Saved results to:", out_dir)
print("EXTRACTION COMPLETED")
