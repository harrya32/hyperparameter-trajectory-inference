import argparse
import os
from typing import List

import torch


def quantile_to_suffix(quantile: float) -> str:
    return str(quantile).replace(".", "")


def parse_quantiles(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-quantile HTI tensors into a single stacked tensor."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="hti_data",
        help="Directory containing hti_data_q*.pt files.",
    )
    parser.add_argument(
        "--quantiles",
        type=str,
        default="0.01,0.1,0.25,0.5,0.75,0.9,0.99",
        help="Comma-separated quantiles in the stack order.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="hti_data/hti_data_combined.pt",
        help="Output path for stacked tensor.",
    )
    parser.add_argument(
        "--nlot_output_path",
        type=str,
        default="../NLOT/data/quantile_data_new.pt",
        help="Optional path to also copy the combined tensor for NLOT training.",
    )
    args = parser.parse_args()

    quantiles = parse_quantiles(args.quantiles)
    tensors = []

    for q in quantiles:
        suffix = quantile_to_suffix(q)
        fp = os.path.join(args.input_dir, f"hti_data_q{suffix}.pt")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing quantile file: {fp}")
        tensor = torch.load(fp, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor at {fp}, got {type(tensor)}")
        tensors.append(tensor)

    first_shape = tensors[0].shape
    for i, tensor in enumerate(tensors[1:], start=1):
        if tensor.shape != first_shape:
            raise ValueError(
                f"Shape mismatch at quantile index {i}: {tensor.shape} vs {first_shape}"
            )

    combined = torch.stack(tensors, dim=0)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(combined, args.output_path)
    print(f"Saved combined HTI tensor to {args.output_path} with shape {tuple(combined.shape)}")

    if args.nlot_output_path:
        nlot_dir = os.path.dirname(args.nlot_output_path)
        if nlot_dir:
            os.makedirs(nlot_dir, exist_ok=True)
        torch.save(combined, args.nlot_output_path)
        print(f"Copied combined HTI tensor to {args.nlot_output_path}")


if __name__ == "__main__":
    main()
