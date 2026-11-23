
import faiss
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
BASE = Path("/scratch-shared/scur0915/embedding_analysis")
FAISS_DIR = Path("/scratch-shared/scur0915/msmarco-sbert")
OUT = BASE / "dl19_all_queries_all_seeds"

# Load missing document IDs
missing_df = pd.read_csv(
    OUT / "missing_docids.csv",
    dtype={"doc_id": str}
)

missing_ids = set(missing_df.doc_id)

print("Missing embeddings:", len(missing_ids))

# Locate document positions in FAISS
positions = {}
new_docids = []
new_embeddings = None

if missing_ids:

    print("\nLocating documents in FAISS...")

    with open(FAISS_DIR / "docid") as f:
        for position, line in enumerate(f):

            doc_id = line.strip()

            if doc_id in missing_ids:
                positions[doc_id] = position

            if len(positions) == len(missing_ids):
                break

    print("Documents found:", len(positions))

    not_found = missing_ids - set(positions)

    if not_found:
        raise ValueError(
            f"Documents missing from FAISS: {not_found}"
        )

    # Load full FAISS index
    print("\nLoading FAISS index...")

    faiss.omp_set_num_threads(4)

    index = faiss.read_index(
        str(FAISS_DIR / "index")
    )

    print("Index size:", index.ntotal)
    print("Embedding dimensions:", index.d)

    # Extract missing embeddings
    new_docids = sorted(positions)

    new_embeddings = np.array([
        index.reconstruct(positions[d])
        for d in new_docids
    ], dtype=np.float32)

    print("Extracted shape:", new_embeddings.shape)

    # Validate embeddings
    if not np.isfinite(new_embeddings).all():
        raise ValueError("Invalid embedding values")

    norms = np.linalg.norm(new_embeddings, axis=1)

    if np.any(norms == 0):
        raise ValueError("Zero-length embedding")

    print("Minimum norm:", norms.min())
    print("Maximum norm:", norms.max())

    # Save additional embeddings
    np.savez(
        OUT / "additional_embeddings.npz",
        docids=np.array(new_docids),
        embeddings=new_embeddings
    )

    del index

# Load existing embeddings
with np.load(BASE / "ground_truth_embeddings.npz") as data:

    old_docids = data["docids"]
    old_embeddings = data["embeddings"]

# Combine embeddings
if new_embeddings is not None:

    all_docids = np.concatenate([
        old_docids,
        np.array(new_docids)
    ])

    all_embeddings = np.concatenate([
        old_embeddings,
        new_embeddings
    ], axis=0)

else:

    all_docids = old_docids
    all_embeddings = old_embeddings

# Verify retrieval coverage
retrieved_df = pd.read_csv(
    OUT / "retrieved_docids.csv",
    dtype={"doc_id": str}
)

retrieved_ids = set(retrieved_df.doc_id)

remaining = retrieved_ids - set(all_docids)

if remaining:
    raise ValueError(
        f"Remaining missing embeddings: {remaining}"
    )

# Save combined embeddings
np.savez(
    OUT / "combined_embeddings.npz",
    docids=all_docids,
    embeddings=all_embeddings
)

print("\nFINAL SUMMARY ")
print("Original embeddings:", len(old_docids))
print("Additional embeddings:", len(new_docids))
print("Combined embeddings:", len(all_docids))
print("Combined shape:", all_embeddings.shape)
print("Retrieved documents:", len(retrieved_ids))
print("Remaining missing embeddings:", len(remaining))
print("Embedding coverage: 100%")

print("\nSaved to:", OUT)
print("EXTRACTION COMPLETED")
