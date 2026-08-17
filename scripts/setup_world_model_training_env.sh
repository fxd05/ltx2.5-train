#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

bash scripts/setup_world_model_env.sh
venv_python="$repo_root/.venv/bin/python"

site_packages="$($venv_python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
cat > "$site_packages/ltx2_local_sources.pth" <<EOF
$repo_root/packages/ltx-core/src
$repo_root/packages/ltx-pipelines/src
$repo_root/packages/ltx-trainer/src
EOF

if ! "$venv_python" - <<'PY'
import bitsandbytes
import imageio
import optimum
import pandas
import peft
import pydantic
import wandb
PY
then
    "$venv_python" -m pip install \
        --upgrade-strategy only-if-needed \
        --prefer-binary \
        --no-compile \
        "bitsandbytes>=0.45.2" "google-genai>=2.0" "imageio>=2.37.0" "imageio-ffmpeg>=0.6.0" \
        "openai>=2.0" "opencv-python-headless>=4.11.0.86" "optimum-quanto>=0.2.6" "pandas>=2.2.3" \
        "peft>=0.14.0" "pillow-heif>=0.21.0" "pydantic>=2.10.4" "rich>=13.9.4" \
        "scenedetect>=0.6.5.2" "sentencepiece>=0.2.0" "soundfile>=0.12.1" "typer>=0.15.1" "wandb>=0.27.0"
fi

"$venv_python" - <<'PY'
import torch
from ltx_trainer.config import LtxTrainerConfig

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"LtxTrainerConfig={LtxTrainerConfig.__name__}")
PY

echo "Training environment ready: $venv_python"
