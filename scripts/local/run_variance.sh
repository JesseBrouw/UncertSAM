#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/local/setup_env.sh"
setup

cd "$PROJECT_ROOT"

python src/variance_network.py \
    --config-name local_config \
    experiment=uncertainty-quantification/training-fitting/variance_network_sam2_tiny
