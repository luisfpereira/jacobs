#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$HERE/../config.sh"

cd "$HERE"

"$PY_DEFORMETRICA" -m to_global
"$PY_VARIFOLD" -m post_dists
"$PY_DEFORMETRICA" -m post_dists_deformetrica
