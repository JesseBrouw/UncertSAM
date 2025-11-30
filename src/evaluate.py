from src.cli.eval import main
from src.uncertsam2.evaluation import (
    decode_mask,
    evaluate_dataset,
    load_targets,
    predict_batch,
    upscale_and_binarise,
)

__all__ = [
    "decode_mask",
    "evaluate_dataset",
    "load_targets",
    "main",
    "predict_batch",
    "upscale_and_binarise",
]

if __name__ == "__main__":
    main()