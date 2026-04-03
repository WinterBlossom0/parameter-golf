#!/bin/bash
# RunPod 8xH100 training script — full competition stack
# Usage: Copy train_gpt.py + data to RunPod, then: bash run_runpod.sh
#
# Prerequisites on RunPod:
#   - PyTorch >= 2.3 with CUDA (most RunPod PyTorch templates have this)
#   - pip install sentencepiece numpy
#   - Dataset at ./data/datasets/fineweb10B_sp1024/
#   - Tokenizer at ./data/tokenizers/fineweb_1024_bpe.model

set -euo pipefail

pip install sentencepiece 2>/dev/null || true

# All defaults are baked into train_gpt.py — just set paths + seed + wallclock.
# The script auto-detects 8 GPUs via torchrun.

run_seed() {
    local seed=$1
    local run_id="fullstack_s${seed}"
    echo "=== Starting ${run_id} ==="
    RUN_ID=${run_id} \
    SEED=${seed} \
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
    MAX_WALLCLOCK_SECONDS=600 \
    OMP_NUM_THREADS=1 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py
    echo "=== ${run_id} complete ==="
    echo "Log: logs/${run_id}.txt"
    echo "Artifact: final_model.int6.ptz"
    # Save artifact per seed
    cp final_model.int6.ptz "final_model.int6.${seed}.ptz" 2>/dev/null || true
    echo ""
}

# Run 3 seeds for reproducibility
run_seed 1337
run_seed 42
run_seed 2025

echo "=== All runs complete ==="
echo "Check logs/ for results. Pick the best seed for submission."
