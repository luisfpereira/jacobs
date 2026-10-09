#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$HERE/../config.sh"

cd "$HERE"


# "$PY_POLPO" -m rank_selection_diagnostics
# "$PY_POLPO" -m rank_selection_diagnostics_viz
"$PY_POLPO" -m rank_selection
# "$PY_POLPO" -m rank_selection_viz
