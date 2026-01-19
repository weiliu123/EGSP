# src/egsp/scfoundation_bridge.py
import os
import sys
from pathlib import Path
import torch

def _try_import():
    from model.load import load_model_frommmf, gatherData
    return load_model_frommmf, gatherData

# def import_scfoundation():
#     """
#     Try importing scFoundation's `model.load`.
#     Users can set env var SCFOUNDATION_MODEL_DIR to point to .../scFoundation/model
#     (the directory that contains `model/` or `load.py` depending on their repo layout).
#     """
#     try:
#         return _try_import()
#     except ModuleNotFoundError:
#         pass
#
#     sc_dir = os.environ.get("SCFOUNDATION_MODEL_DIR", "E:/deps/scFoundation/")
#     if sc_dir:
#         scf_root = Path(sc_dir)
#
#         # 关键：同时加入 root 和 root/model
#         sys.path.insert(0, str(scf_root))
#         sys.path.insert(0, str(scf_root / "model"))
#
#         return _try_import()
#
#     raise ModuleNotFoundError(
#         "Cannot import scFoundation. Please set environment variable "
#         "SCFOUNDATION_MODEL_DIR to your scFoundation 'model' directory, e.g.\n"
#         "Windows PowerShell:\n"
#         "  $env:SCFOUNDATION_MODEL_DIR = 'E:/deps/scFoundation'\n"
#         "Or add that path to sys.path before importing EGSP_End2End.\n."
#     )

def import_scfoundation():
    """
    Try importing scFoundation's `model.load`.

    Users can set env var SCFOUNDATION_MODEL_DIR to point to the scFoundation root
    directory (the one containing `model/`).

    This function returns:
        - load_model_frommmf_safe: device-aware model loader (CPU/GPU)
        - gatherData: original scFoundation function
    """
    try:
        load_model_frommmf, gatherData = _try_import()
    except ModuleNotFoundError:
        sc_dir = os.environ.get("SCFOUNDATION_MODEL_DIR", "E:/deps/scFoundation/")
        if not sc_dir:
            raise

        scf_root = Path(sc_dir)

        # Key: add both root and root/model to sys.path
        sys.path.insert(0, str(scf_root))
        sys.path.insert(0, str(scf_root / "model"))

        load_model_frommmf, gatherData = _try_import()

    # -----------------------------
    # ✅ Device-aware wrapper
    # -----------------------------
    def load_model_frommmf_safe(ckpt_path):
        model, config = load_model_frommmf(ckpt_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        return model, config

    return load_model_frommmf_safe, gatherData
