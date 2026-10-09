#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$HERE/../config.sh"

cd "$HERE"


# "$PY_POLPO" -m pca_reconstruction_viz
"$PY_POLPO" -m pairplot


