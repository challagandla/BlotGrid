#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="python3"
with_ml=0
with_sam2=0
ml_runtime="auto"

usage() {
    echo "Usage: scripts/setup_ubuntu.sh [--with-ml] [--with-sam2]"
    echo "                               [--ml-runtime auto|cu118|cpu]"
    echo "                               [--python /path/to/python]"
    echo
    echo "Creates .venv and installs BlotGrid. --with-ml installs the optional"
    echo "tested PyTorch/TorchVision pair. --with-sam2 also installs the pinned"
    echo "experimental SAM 2.1 ROI refiner and implies --with-ml."
}

while (($#)); do
    case "$1" in
        --with-ml)
            with_ml=1
            shift
            ;;
        --with-sam2)
            with_sam2=1
            with_ml=1
            shift
            ;;
        --ml-runtime)
            if (($# < 2)); then
                echo "error: --ml-runtime needs auto, cu118, or cpu" >&2
                exit 2
            fi
            ml_runtime="$2"
            shift 2
            ;;
        --python)
            if (($# < 2)); then
                echo "error: --python needs an interpreter path" >&2
                exit 2
            fi
            python_bin="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "${ml_runtime}" != "auto" && "${ml_runtime}" != "cu118" && "${ml_runtime}" != "cpu" ]]; then
    echo "error: --ml-runtime must be auto, cu118, or cpu" >&2
    exit 2
fi
if [[ "${ml_runtime}" != "auto" ]] && ((!with_ml)); then
    echo "error: --ml-runtime requires --with-ml" >&2
    exit 2
fi

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "error: Python interpreter not found: ${python_bin}" >&2
    exit 1
fi

cd "${repo_root}"

if ! "${python_bin}" -m venv .venv; then
    echo >&2
    echo "Could not create .venv. On Ubuntu install the venv package first:" >&2
    echo "  sudo apt update && sudo apt install python3-venv" >&2
    exit 1
fi

venv_python="${repo_root}/.venv/bin/python"
"${venv_python}" -m pip install --upgrade pip wheel
"${venv_python}" -m pip install -r requirements.txt

if ((with_ml)); then
    selected_runtime="${ml_runtime}"
    if [[ "${selected_runtime}" == "auto" ]]; then
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
            selected_runtime="cu118"
        else
            selected_runtime="cpu"
        fi
    fi
    pytorch_index="https://download.pytorch.org/whl/${selected_runtime}"
    ml_install_flags=(--upgrade)
    # Pip considers 2.7.1+cpu and 2.7.1+cu118 equal to a public ==2.7.1
    # requirement. Detect the local runtime explicitly and force replacement
    # only when the requested CPU/CUDA build or matched package version differs.
    if ! "${venv_python}" - "${selected_runtime}" <<'PY'
from importlib import metadata
from pathlib import Path
import sys

expected = {}
for line in Path("requirements-ml.txt").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "==" in line:
        package, version = line.split("==", 1)
        expected[package.strip()] = version.strip()
try:
    import torch
    import torchvision
except (ImportError, OSError):
    raise SystemExit(1)

versions_match = (
    metadata.version("torch").split("+", 1)[0] == expected.get("torch")
    and metadata.version("torchvision").split("+", 1)[0]
    == expected.get("torchvision")
)
runtime = sys.argv[1]
runtime_matches = (
    torch.version.cuda is None
    if runtime == "cpu"
    else str(torch.version.cuda) == "11.8"
)
raise SystemExit(0 if versions_match and runtime_matches else 1)
PY
    then
        ml_install_flags+=(--force-reinstall)
    fi
    "${venv_python}" -m pip install "${ml_install_flags[@]}" \
        --index-url "${pytorch_index}" \
        -r requirements-ml.txt
    "${venv_python}" -m pip check
    "${venv_python}" - "${selected_runtime}" <<'PY'
import sys

import torch
import torchvision

runtime = sys.argv[1]
print(f"PyTorch {torch.__version__}; TorchVision {torchvision.__version__}")
print(f"CUDA build: {torch.version.cuda}; CUDA available: {torch.cuda.is_available()}")
if runtime == "cu118" and not torch.cuda.is_available():
    raise SystemExit(
        "CUDA wheels installed, but PyTorch cannot access a GPU. "
        "Check nvidia-smi outside containers/sandboxes or use --ml-runtime cpu."
    )
if torch.cuda.is_available():
    print(f"Training device: {torch.cuda.get_device_name(0)}")
else:
    print("Training device: CPU")
PY
fi

if ((with_sam2)); then
    SAM2_BUILD_CUDA=0 "${venv_python}" -m pip install \
        -r requirements-sam2.txt
fi

mkdir -p "${repo_root}/models"

echo
echo "BlotGrid is ready."
echo "Start it with: ${repo_root}/run_blotgrid.sh"
if ((!with_ml)); then
    echo "To add AI training later: scripts/setup_ubuntu.sh --with-ml"
fi
if ((with_sam2)); then
    echo "Download SAM weights with:"
    echo "  .venv/bin/python scripts/download_pretrained_models.py --model sam2.1-hiera-small"
fi
