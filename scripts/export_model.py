#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export an abliterated model: apply best trial steering and push to HuggingFace.

Usage:
    python scripts/export_model.py \
        --model google/gemma-4-31B-it \
        --checkpoint checkpoints_gemma4_31b_v8 \
        --trial 13 \
        --config configs/gemma4_31b_v8_direct.toml \
        --push-to wangzhang/gemma-4-31B-it-abliterated
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Base HF model ID")
    parser.add_argument("--checkpoint", required=True, help="Optuna checkpoint dir")
    parser.add_argument(
        "--trial", type=int, required=True, help="Trial number to export"
    )
    parser.add_argument("--config", required=True, help="Config TOML path")
    parser.add_argument(
        "--push-to", default=None, help="HF repo to push to (omit to skip upload)"
    )
    parser.add_argument(
        "--save-local", default=None, help="Also save locally to this path"
    )
    parser.add_argument(
        "--format",
        choices=("merged", "adapter"),
        default="merged",
        help=(
            "Export a standalone merged checkpoint or the exact dynamic LoRA "
            "adapter (avoids BF16 merge-rounding drift)."
        ),
    )
    parser.add_argument(
        "--emit-fp4-plan",
        default=None,
        metavar="PLAN.pt",
        help=(
            "Record the best trial's resolved direct/EGA edits to this .pt "
            "file (does not mutate weights or save a model). Replay it against "
            "a native-FP4 checkpoint with `abliterix-abliterate-fp4 <fp4_dir> "
            "PLAN.pt <out_dir>` to bake a compact abliterated FP4 model instead "
            "of a full BF16 merge. Skips the normal merged/adapter export."
        ),
    )
    args = parser.parse_args()

    os.environ["AX_CONFIG"] = args.config
    sys.argv = [
        sys.argv[0],
        "--model.model-id",
        args.model,
        "--inference.batch-size",
        "4",
    ]

    import torch

    torch.set_grad_enabled(False)

    from abliterix.scriptlib import (
        apply_trial_artifact,
        compute_trial_vectors,
        extract_trial_artifact,
        load_trial,
        setup_io,
    )
    from abliterix.core.engine import SteeringEngine
    from abliterix.core.steering import apply_steering
    from abliterix.data import load_prompt_dataset
    from abliterix.settings import AbliterixConfig
    from abliterix.util import flush_memory

    setup_io()

    # Load trial
    trial = load_trial(args.checkpoint, args.model, args.trial)
    artifact = extract_trial_artifact(trial)
    direction_index = artifact.vector_index
    parameters = artifact.profiles
    routing = artifact.routing
    refusals = trial.user_attrs.get("refusals")
    kl = trial.user_attrs.get("kl_divergence")
    print(f"Trial #{args.trial}: refusals={refusals}, KL={kl}")

    # Load model
    config = AbliterixConfig()
    apply_trial_artifact(config, artifact)
    engine = SteeringEngine(config)

    # Compute steering vectors
    print("\nComputing steering vectors...")
    benign = load_prompt_dataset(config, config.benign_prompts)
    target = load_prompt_dataset(config, config.target_prompts)
    benign_states = engine.extract_hidden_states_batched(benign)
    target_states = engine.extract_hidden_states_batched(target)
    vectors = compute_trial_vectors(artifact, benign_states, target_states, config)

    # Profile MoE experts if applicable
    safety_experts = None
    if engine.has_expert_routing():
        print("Profiling MoE experts...")
        safety_experts = engine.identify_safety_experts(benign, target)

    del benign, target
    flush_memory()

    # FP4 plan-only path: record the resolved edits and stop, so they can be
    # replayed offline against the FP4 checkpoint (no BF16 merge / upload).
    if args.emit_fp4_plan is not None:
        from abliterix.core.fp4_repack import (
            record_steering_plan_from_trial,
            save_plan,
        )

        print(f"\nRecording FP4 steering plan to {args.emit_fp4_plan}...")
        edits = record_steering_plan_from_trial(
            engine,
            vectors,
            direction_index,
            parameters,
            config,
            benign_states=benign_states,
            target_states=target_states,
        )
        save_plan(edits, args.emit_fp4_plan)
        n_ega = sum(1 for e in edits if e.kind == "ega")
        n_direct = sum(1 for e in edits if e.kind == "direct")
        print(
            f"Plan saved: {n_ega} EGA + {n_direct} direct edits. Bake with:\n"
            f"  abliterix-abliterate-fp4 <fp4_snapshot_dir> "
            f"{args.emit_fp4_plan} <out_dir>"
        )
        return

    # Apply steering
    print("Applying steering (direct weight editing)...")
    apply_steering(
        engine,
        vectors,
        direction_index,
        parameters,
        config,
        safety_experts=safety_experts,
        routing_config=routing,
        benign_states=benign_states,
        target_states=target_states,
    )
    print("Steering applied.")
    del benign_states, target_states
    flush_memory()

    # Save locally first (root filesystem is tiny, use /workspace)
    save_dir = args.save_local or "/workspace/export_model"
    print(f"\nSaving {args.format} export to {save_dir}...")
    if args.format == "adapter":
        # Adapter export preserves the live LoRA computation. A BF16 merge can
        # introduce small rounding drift even though top-1 output is stable.
        engine.export_adapter(save_dir)
    else:
        # Use the engine's export contract so runtime-only and unfaithful
        # quantized exports fail loudly instead of silently losing steering.
        model = engine.export_merged()
        model.save_pretrained(save_dir)
    engine.tokenizer.save_pretrained(save_dir)
    print("Local save complete.")

    if args.push_to is None:
        print(f"Local-only export complete at {save_dir}. Skipping HF upload.")
        return

    # Push to HuggingFace
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Pushing to {args.push_to}...")
    api.create_repo(args.push_to, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=save_dir,
        repo_id=args.push_to,
        repo_type="model",
    )
    print(f"Done! Model pushed to https://huggingface.co/{args.push_to}")


if __name__ == "__main__":
    main()
