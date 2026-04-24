"""Print Qwen3.5 kernel-facing shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastkernels.models import Qwen35MoESpec, qwen35_35b_a3b_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, help="Optional Hugging Face config.json")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128])
    args = parser.parse_args()

    spec = Qwen35MoESpec.from_json_file(args.config) if args.config else qwen35_35b_a3b_spec()
    for line in spec.summary_lines(args.tokens):
        print(line)

    print("\nMoE shapes:")
    for tokens in args.tokens:
        print(f"\n[tokens={tokens}]")
        for name, shape in spec.moe_shapes(tokens).items():
            print(f"{name}: {shape}")


if __name__ == "__main__":
    main()
