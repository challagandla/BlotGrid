#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${BLOTGRID_PYTHON:-${repo_root}/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
    echo "BlotGrid environment not found: ${python_bin}" >&2
    echo "Run scripts/setup_ubuntu.sh first." >&2
    exit 1
fi

cd "${repo_root}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${repo_root}/.cache}"
exec "${python_bin}" -m streamlit run app.py "$@"
