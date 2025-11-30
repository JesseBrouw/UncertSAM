#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/local/setup_env.sh"
setup

python "$PROJECT_ROOT/src/prompt_tuning.py" \
    --config-name local_config \
    experiment=refinement/training-fitting/prompt_tuning_variance_sam2_tiny
