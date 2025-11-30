#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/local/setup_env.sh"
setup

python "$PROJECT_ROOT/src/laplace_approximation.py" \
    --config-name local_config \
    experiment=uncertainty-quantification/training-fitting/laplace_sam2_tiny
