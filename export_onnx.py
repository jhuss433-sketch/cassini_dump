"""
Export a trained BowShockTCN checkpoint to ONNX.

Usage:
    python export_onnx.py --checkpoint checkpoints/best_model.pt --output checkpoints/best_model.onnx
"""

import argparse
from pathlib import Path

import torch

from tcn_model import BowShockTCN


# ---------------------------------------------------------------------------
# Model loading (mirrors inference.py's load_model)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, num_energy_bins: int) -> BowShockTCN:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})

    model = BowShockTCN(
        num_energy_bins = saved_args.get("num_energy_bins", num_energy_bins),
        num_channels    = saved_args.get("num_channels", [32, 64, 128, 64, 32]),
        dropout         = saved_args.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(
    model: BowShockTCN,
    output_path: Path,
    num_energy_bins: int,
    time_steps: int,
    opset: int,
    dynamic_axes: bool,
):
    dummy_input = torch.randn(1, time_steps, num_energy_bins)

    axes = None
    if dynamic_axes:
        axes = {
            "input":  {0: "batch", 1: "time_steps"},
            "logits": {0: "batch", 1: "time_steps"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=axes,
        opset_version=opset,
    )
    print(f"Saved ONNX model to {output_path}")


def verify_onnx(output_path: Path, num_energy_bins: int, time_steps: int):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print('onnx onnxruntime not installed  skipping verification ')
        return

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(output_path))
    dummy_input = torch.randn(1, time_steps, num_energy_bins).numpy()
    session.run(None, {"input": dummy_input})
    print("ONNX model check + inference passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export a BowShockTCN checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--output", default=None, help="Output .onnx path (defaults to checkpoint name)")
    parser.add_argument("--num-energy-bins", type=int, default=63, help="Fallback if not in checkpoint args")
    parser.add_argument("--time-steps", type=int, default=128, help="Sequence length used to trace the model")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-dynamic-axes", action="store_true", help="Fix batch/time_steps size in the graph")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output) if args.output else checkpoint_path.with_suffix(".onnx")

    if output_path.is_dir():
        output_path = output_path / checkpoint_path.with_suffix(".onnx").name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output path: {output_path.resolve()}")

    model = load_model(checkpoint_path, args.num_energy_bins)
    num_energy_bins = model.tcn[0].block[0].in_channels
    print(f"Loaded model from {checkpoint_path} (energy_bins={num_energy_bins})")

    export_onnx(
        model,
        output_path,
        num_energy_bins=num_energy_bins,
        time_steps=args.time_steps,
        opset=args.opset,
        dynamic_axes=not args.no_dynamic_axes,
    )
    verify_onnx(output_path, num_energy_bins, args.time_steps)


if __name__ == "__main__":
    main()
