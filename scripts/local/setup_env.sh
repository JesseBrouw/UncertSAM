#!/bin/bash

setup() {
    export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
    export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf-cache}"

    if [ -f "$PROJECT_ROOT/configs/.secrets" ]; then
        source "$PROJECT_ROOT/configs/.secrets"
    fi

    if [ -d "$PROJECT_ROOT/.venv" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    else
        echo ".venv not found; activate your environment manually" >&2
    fi
}
