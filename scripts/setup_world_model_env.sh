#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
venv_dir="$repo_root/.venv"

if ! "$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("The selected Python does not expose a CUDA-enabled PyTorch installation.")
print(f"Using system PyTorch {torch.__version__} with CUDA {torch.version.cuda}")
PY
then
    echo "Set PYTHON_BIN to the server Python that contains the preinstalled CUDA PyTorch." >&2
    exit 1
fi

# LTX-2.5 is large and the server already provides a CUDA-enabled PyTorch build.
# Reuse that build instead of resolving/downloading another torch wheel through uv.
if [[ ! -x "$venv_dir/bin/python" ]] || ! "$venv_dir/bin/python" -c "import torch"; then
    "$python_bin" -m venv --clear --system-site-packages "$venv_dir"
fi
venv_python="$venv_dir/bin/python"

if ! "$venv_python" - <<'PY'
import OpenImageIO
import accelerate
import av
import cloudpickle
import transformers
PY
then
    "$venv_python" -m pip install \
        --upgrade-strategy only-if-needed \
        --prefer-binary \
        --no-compile \
        "av" "cloudpickle>=3.1" "openimageio" "accelerate" "transformers>=5.8.0,<5.15"
fi

site_packages="$($venv_python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
cat > "$site_packages/ltx2_local_sources.pth" <<EOF
$repo_root/packages/ltx-core/src
$repo_root/packages/ltx-pipelines/src
EOF

"$venv_python" - <<'PY'
import av
import torch
from ltx_pipelines.ic_lora import ICLoraPipeline

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu_count={torch.cuda.device_count()}")
    print(f"gpu_0={torch.cuda.get_device_name(0)}")
print(f"pyav={av.__version__}")
print(f"ICLoraPipeline={ICLoraPipeline.__name__}")
PY

echo "Environment ready: $venv_python"
