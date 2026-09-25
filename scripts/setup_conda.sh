#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="${1:-resolve}"

conda create -y -n "$ENV_NAME" python=3.13 numpy scipy pip
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n "$ENV_NAME" python -m pip install "transformers==4.57.6" "peft==0.20.0" "scikit-learn==1.9.0" pytest
conda run -n "$ENV_NAME" python -m pip install -e "$ROOT"
echo "ready: conda activate $ENV_NAME"
