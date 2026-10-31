#!/bin/bash
# Summary of your queue per model / variant / seed: bash snellius/queue.sh
squeue -u $USER --noheader --format="%T %200j" \
 | sed -e 's/tqe_//' -e 's/DeepSeek-R1-Distill-Qwen-//' \
 | awk '{ st=$1; n=$2
         sub(/^(earth_science|sustainable_living|[a-z0-9]+)_/, "", n)
         c[n" "st]++; k[n]=1; tot[st]++ }
       END { printf "%-42s %8s %8s\n", "model_variant_seed", "RUNNING", "PENDING"
             for (x in k) printf "%-42s %8d %8d\n", x, c[x" RUNNING"], c[x" PENDING"]
             printf "%-42s %8d %8d\n", "TOTAL", tot["RUNNING"], tot["PENDING"] }' | sort
