#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_DIR="${1:-$PROJECT_ROOT/checkpoints}"

mkdir -p "$CHECKPOINT_DIR"
echo "Saving checkpoints under: $CHECKPOINT_DIR"

if command -v wget >/dev/null 2>&1; then
    DL_CMD=(wget -nv -P "$CHECKPOINT_DIR")
elif command -v curl >/dev/null 2>&1; then
    DL_CMD=(curl -L -o)
else
    echo "Please install wget or curl to download checkpoints." >&2
    exit 1
fi

sam_download() {
    local url="$1"
    local target="$CHECKPOINT_DIR/$(basename "$url")"
    if [[ -f "$target" ]]; then
        echo "Skipping existing file: $target"
        return
    fi

    echo "Downloading $(basename "$url")"
    if [[ "${DL_CMD[0]}" == "wget" ]]; then
        "${DL_CMD[@]}" "$url"
    else
        "${DL_CMD[@]}" "$target" "$url"
    fi
}

SAM_VIT_H_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_VIT_L_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
SAM_VIT_B_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824"
SAM2_HIERA_T_URL="${SAM2_BASE_URL}/sam2_hiera_tiny.pt"
SAM2_HIERA_S_URL="${SAM2_BASE_URL}/sam2_hiera_small.pt"
SAM2_HIERA_B_URL="${SAM2_BASE_URL}/sam2_hiera_base_plus.pt"
SAM2_HIERA_L_URL="${SAM2_BASE_URL}/sam2_hiera_large.pt"

SAM2P1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2P1_HIERA_T_URL="${SAM2P1_BASE_URL}/sam2.1_hiera_tiny.pt"
SAM2P1_HIERA_S_URL="${SAM2P1_BASE_URL}/sam2.1_hiera_small.pt"
SAM2P1_HIERA_B_URL="${SAM2P1_BASE_URL}/sam2.1_hiera_base_plus.pt"
SAM2P1_HIERA_L_URL="${SAM2P1_BASE_URL}/sam2.1_hiera_large.pt"

# Uncomment the checkpoints you actually need.
# sam_download "$SAM_VIT_B_URL"
# sam_download "$SAM_VIT_L_URL"
# sam_download "$SAM_VIT_H_URL"

# sam_download "$SAM2_HIERA_T_URL"
# sam_download "$SAM2_HIERA_S_URL"
# sam_download "$SAM2_HIERA_B_URL"
# sam_download "$SAM2_HIERA_L_URL"

sam_download "$SAM2P1_HIERA_T_URL"
# sam_download "$SAM2P1_HIERA_S_URL"
# sam_download "$SAM2P1_HIERA_B_URL"
# sam_download "$SAM2P1_HIERA_L_URL"

echo "Done."
