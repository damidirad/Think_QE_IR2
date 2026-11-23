## Ground-Truth Qrels: Data Provenance

### 1. Source of relevance judgments

The TREC Deep Learning 2019 (DL19) and 2020 (DL20) benchmarks provide relevance judgments (*qrels*) indicating how relevant individual MS MARCO passages are to particular queries.

The original judgments come from TREC/NIST. Castorini distributes copies through its Anserini Tools repository, which Pyserini uses to access the evaluation data.

### 2. Data provenance

```text
TREC Deep Learning 2019 / 2020
    │
    │ Original relevance judgments
    ▼
Castorini / Anserini Tools (GitHub)
    │
    │ Downloaded by snellius/01_download.sh
    ▼
Pyserini / Anserini local cache
    │
    ├── qrels.dl19-passage.txt
    └── qrels.dl20-passage.txt
    │
    │ Read and combined into a CSV
    ▼
ground_truth_qrels.csv
    │
    ▼
Gold-document reachability and
retrieval-failure analysis
```

**Castorini repository:**

https://github.com/castorini/anserini-tools

**Local Pyserini cache:**

```text
/home/scur0915/.cache/anserini/topics-and-qrels/
```

**Combined output:**

```text
/scratch-shared/scur0915/embedding_analysis/ground_truth_qrels.csv
```

### 3. CSV structure

The combined CSV contains five columns:

| Column | Description |
|---|---|
| `query_id` | TREC query identifier |
| `unused` | Placeholder field (`Q0` or `0`) |
| `doc_id` | MS MARCO passage identifier |
| `relevance` | Graded relevance judgment |
| `dataset` | `dl19` or `dl20` |

A relevance score of `0` means the passage was judged nonrelevant. Positive scores indicate relevant passages, with larger values indicating greater relevance.

The CSV includes all available judgments, not only relevant documents.

### 4. Validation results

The combined CSV was compared against the qrels files resolved by Pyserini 0.22.1.

| Dataset | Cached judgments | CSV judgments | Exact match |
|---|---:|---:|---|
| DL19 | 9,260 | 9,260 | Yes |
| DL20 | 11,386 | 11,386 | Yes |
| **Total** | **20,646** | **20,646** | **Yes** |

The comparison found zero missing and zero extra judgments in either dataset.

**Conclusion:** The combined CSV matches the cached relevance judgments used in the ThinkQE environment.

### 5. Important distinction: Qrels vs. embeddings

The qrels contain relevance labels, not semantic embeddings.

- `ground_truth_qrels.csv`: Query-document relevance judgments from TREC. 
- `ground_truth_embeddings.npz`: Vector representations extracted from a separate FAISS index for document IDs present in the qrels. Contains vectors for document IDs appearing in the DL19/DL20 qrels, including relevance-0 documents.
- `additional_embeddings.npz`: Vectors for retrieved documents not already present in the ground-truth embedding subset. Contains vectors for retrieved passages that were missing from the first embedding file.
- `combined_embeddings.npz`: Ground-truth and additional vectors combined for semantic similarity analysis. Contains both groups of embeddings to compare the semantic similarity of retrieved passages across ThinkQE rounds.



#Golden documents => relevance >=2