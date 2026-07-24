# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline FP4 edit-and-repack CLI.

Replays a recorded steering plan against a native-FP4 checkpoint (MXFP4 /
NVFP4), producing a standalone abliterated **FP4** model — no BF16 expansion.
Only the edited tensors (fused expert ``down_proj``, ``attn.o_proj``, dense
``mlp.down_proj``) are dequantised → projected → re-packed; everything else is
copied through. Peak memory is one layer's expert block, so a 284B FP4 model
bakes on a single GPU instead of the ~568GB / 4× B200 a BF16 merge needs.

Produce the plan during export with ``scripts/export_model.py --emit-fp4-plan
plan.pt`` (it records what the best trial would edit without mutating weights).

Usage
-----
    python -m abliterix.scripts.abliterate_fp4 \\
        /workspace/dsv4_flash_fp4_snapshot \\
        plan.pt \\
        /workspace/dsv4_flash_abliterated_fp4

Notes
-----
* The output keeps ``quantization_config`` — it is still FP4 and loads in any
  NVFP4/MXFP4-capable stack (vLLM native) with no special support.
* MXFP4/NVFP4 is a 4-bit format: re-quantising an edited tensor carries ~10-15%
  mean relative error (the regime the base weights already live in). The tool
  prints the worst per-tensor requant error — validate downstream KL on a small
  model (gpt-oss-20b MXFP4) before trusting it on a 284B target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m abliterix.scripts.abliterate_fp4",
        description=(
            "Replay a steering plan against a native-FP4 checkpoint and write "
            "a standalone abliterated FP4 copy (no BF16 expansion)."
        ),
    )
    p.add_argument(
        "src",
        type=Path,
        help="Source FP4 model directory (config.json + *.safetensors).",
    )
    p.add_argument(
        "plan",
        type=Path,
        help="Recorded steering plan (.pt) from export_model.py --emit-fp4-plan.",
    )
    p.add_argument("dst", type=Path, help="Destination directory (will be created).")
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU dequant/repack (default: use CUDA if available).",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the per-tensor pack/unpack self-consistency check (faster).",
    )
    p.add_argument(
        "--scale-search",
        type=int,
        default=1,
        metavar="S",
        help=(
            "MSE-optimal MXFP4 block-scale search depth for repacking edited "
            "tensors (default 1). Tries the amax-tight exponent and S tighter "
            "ones per block, keeping the lowest-error scale — cuts 4-bit "
            "requant error. 0 = plain amax scale; higher = slower. NVFP4 "
            "ignores it."
        ),
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-shard output.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.src.is_dir():
        print(f"error: {args.src} is not a directory", file=sys.stderr)
        return 2
    if not (args.src / "config.json").exists():
        print(f"error: {args.src}/config.json not found", file=sys.stderr)
        return 2
    if not args.plan.is_file():
        print(f"error: plan {args.plan} not found", file=sys.stderr)
        return 2

    from ..core.fp4_repack import abliterate_fp4_to_disk, load_plan

    edits = load_plan(args.plan)
    if not args.quiet:
        n_ega = sum(1 for e in edits if e.kind == "ega")
        n_direct = sum(1 for e in edits if e.kind == "direct")
        print(f"Loaded plan: {n_ega} EGA + {n_direct} direct edits")

    try:
        abliterate_fp4_to_disk(
            args.src,
            args.dst,
            edits,
            use_cuda=not args.cpu,
            verify_idempotent=not args.no_verify,
            scale_search=args.scale_search,
            verbose=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
