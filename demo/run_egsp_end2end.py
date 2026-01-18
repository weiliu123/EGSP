# demo/run_egsp_end2end.py
from pathlib import Path
import argparse
import torch

from egsp.pipelines.egsp_end2end import run_end2end_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to scFoundation models.ckpt")
    parser.add_argument(
        "--gene_list_path",
        type=str,
        default=None,
        help="Path to OS_scRNA_gene_index.19264.tsv (default: assets/scfoundation/...)"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    demo_dir = root / "demo" / "EGSP_End2End_demo"

    ckpt_path = Path(args.ckpt_path)
    if args.gene_list_path is None:
        gene_list_path = root / "assets" / "scfoundation" / "OS_scRNA_gene_index.19264.tsv"
    else:
        gene_list_path = Path(args.gene_list_path)

    output_dir = root / "checkpoints" / "EGSP_End2End_tmp"
    use_gpu = bool(args.use_gpu)

    res = run_end2end_demo(
        demo_dir=demo_dir,
        scfoundation_ckpt_path=ckpt_path,
        gene_list_path=gene_list_path,
        output_dir=output_dir,
        epochs=args.epochs,
        use_gpu=use_gpu,
        report_to="none",
    )

    print("\n=== Metrics ===")
    print("Train:", res.train_metrics)
    print("Eval :", res.eval_metrics)
    print("Test :", res.test_metrics)
    print("Test C-index:", res.test_cindex)
    print("Example risk scores (first 10):", res.test_risk[:10])


if __name__ == "__main__":
    main()
