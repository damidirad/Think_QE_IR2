#!/bin/bash
# Build thinkqe_qwen38 without conda downloads: copy thinkqe_qwen3 (python 3.10, Java, pyserini,
# spaCy model, JVM fix), then upgrade to current vLLM. vLLM >= 0.19 needs numpy 2, so every
# compiled package that links against numpy is upgraded too.
set -eo pipefail
module purge; module load 2023 Anaconda3/2023.07-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env remove -y -n thinkqe_qwen38 2>/dev/null || true
conda create -y -n thinkqe_qwen38 --clone thinkqe_qwen3 --offline
conda activate thinkqe_qwen38
pip install --no-cache-dir -r snellius/requirements_qwen38.txt
pip install --no-cache-dir -U "numpy>=2,<2.3" pandas scipy scikit-learn spacy thinc blis
python -c "from pyserini.search.lucene import LuceneSearcher; print('pyserini OK')"
python -c "import spacy; spacy.load('en_core_web_sm'); print('spacy OK')"
python -c "import vllm, transformers, torch, numpy; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__, 'numpy', numpy.__version__)"
pip freeze > snellius/frozen_thinkqe_qwen38.txt
echo "Environment thinkqe_qwen38 ready."
