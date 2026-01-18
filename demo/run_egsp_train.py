# demo/run_egsp_train.py
from pathlib import Path
import torch

from egsp.pipelines.egsp_train import run_egsp_train_demo


def main():
    root = Path(__file__).resolve().parents[1]  # repo root

    demo_dir = root / "demo" / "EGSP_demo"
    ckpt_path = root / "checkpoints" / "egsp_demo.pt"
    output_dir = root / "checkpoints" / "EGSP_tmp"

    use_gpu = True
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print("Using device:", device)

    res = run_egsp_train_demo(
        demo_dir=demo_dir,
        ckpt_path=ckpt_path,
        output_dir=output_dir,
        epochs=10,      # ✅默认 10
        use_gpu=True,   # ✅默认 GPU（有就用）
        seed=42,
        patience=10,
    )

    print("\n=== Metrics ===")
    print("Train:", res.train_metrics)
    print("Eval :", res.eval_metrics)
    print("Test :", res.test_metrics)
    print("Test C-index:", res.test_cindex)
    print("Example risk scores (first 10):", res.test_risk[:10])


if __name__ == "__main__":
    main()
