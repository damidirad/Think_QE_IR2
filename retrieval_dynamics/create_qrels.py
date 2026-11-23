from pathlib import Path
script = r'''#!/usr/bin/env python3
"""Build DL19/DL20 qrels CSV from the same Pyserini qrels used by ThinkQE.

Safe default: refuse to overwrite an existing CSV.
"""
import argparse
import csv
import os
import tempfile
from pathlib import Path

from pyserini.search import get_qrels

DEFAULT_OUTPUT = Path("/scratch-shared/scur0915/embedding_analysis/ground_truth_qrels.csv")
DATASETS = (("dl19", "dl19-passage"), ("dl20", "dl20-passage"))
EXPECTED_COUNTS = {"dl19": 9260, "dl20": 11386}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace an existing CSV after validation")
    args = parser.parse_args()

    output = args.output
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"Existing file preserved: {output}\n"
            "To regenerate it deliberately, rerun with --overwrite."
        )

    all_rows = []
    for dataset, qrels_name in DATASETS:
        # Pyserini resolves cached qrels and may download them if absent.
        qrels = get_qrels(qrels_name)
        rows = []
        seen = set()
        for qid, docs in qrels.items():
            for docid, relevance in docs.items():
                key = (str(qid), str(docid))
                if key in seen:
                    raise ValueError(f"Duplicate qrel in {dataset}: {key}")
                seen.add(key)
                rows.append((str(qid), "0", str(docid), int(relevance), dataset))
        if len(rows) != EXPECTED_COUNTS[dataset]:
            raise ValueError(
                f"{dataset}: expected {EXPECTED_COUNTS[dataset]} judgments "
                f"based on your verified cache, received {len(rows)}. "
                "Refusing to change output."
            )
        rows.sort(key=lambda r: (int(r[0]), int(r[2])))
        all_rows.extend(rows)
        print(f"{dataset}: {len(rows)} judgments, {len(qrels)} queries")

    output.parent.mkdir(parents=True, exist_ok=True)
    # Temporary file is written first so a failure cannot corrupt the target.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["query_id", "unused", "doc_id", "relevance", "dataset"])
            writer.writerows(all_rows)
        if output.exists() and not args.overwrite:
            raise FileExistsError(output)
        os.replace(tmp_name, output)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)

    print(f"Saved {len(all_rows)} judgments to {output}")


if __name__ == "__main__":
    main()
'''
p=Path('/mnt/data/build_ground_truth_qrels.py')
p.write_text(script)
import py_compile
py_compile.compile(str(p), doraise=True)
print(p, "syntax checked")
