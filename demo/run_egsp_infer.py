# demo/run_egsp_infer.py
from pathlib import Path
import torch
from egsp.pipelines.egsp_infer import run_egsp_infer

def main():
    root = Path(__file__).resolve().parents[1]
    demo_dir = root / "demo" / "EGSP_demo"
    ckpt_path = root / "checkpoints" / "egsp_demo.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    risk = run_egsp_infer(demo_dir, ckpt_path, device=device)
    print("Predicted risk scores (first 10):")
    print(risk[:10].squeeze())

if __name__ == "__main__":
    main()
