#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/local/setup_env.sh"
setup

python -m src.cli.eval \
    --config-name local_config \
    experiment=uncertainty-quantification/eval/sam2_tiny
