# Abliterix — SGLang inference backend
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SGLang-backed generation engine with tensor parallelism.

SGLang provides ~29% higher throughput than vLLM on prefix-heavy workloads
thanks to RadixAttention (automatic KV cache reuse across prompts that share
prefixes).  However, RadixAttention is **incompatible with LoRA** and must
be disabled (``--disable-radix-cache``).  Even without RadixAttention, SGLang
is competitive with vLLM thanks to its efficient scheduler and overlap
weight loading for LoRA hot-swap (~78% TTFT reduction).

This module mirrors the :class:`VLLMGenerator` API so callers can use it
interchangeably.  The :class:`SGLangGenerator` wraps SGLang's offline
``Engine`` for batch inference with LoRA adapter hot-swapping.

Requires SGLang >= 0.5.10 for MoE LoRA support.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import Tensor

from ..settings import AbliterixConfig
from ..types import ChatMessage
from ..util import print


# Minimum LoRA rank supported by SGLang (same as vLLM).
_SGLANG_MIN_RANK = 8

# Adapter name used for the steering LoRA — reused across trials.
_ADAPTER_NAME = "abliterix_steering"


class SGLangGenerator:
    """SGLang-backed text generator with LoRA adapter hot-swapping.

    Drop-in replacement for :class:`VLLMGenerator` using SGLang's offline
    ``Engine`` API.  Provides the same generation interface:
    ``generate_text_batched``, ``generate_and_score_batched``,
    ``compute_logprobs_batched``.

    LoRA lifecycle per trial:
      1. ``save_adapter()`` — serialise PEFT-format adapter to tmpfs
      2. ``engine.load_lora_adapter(name, path)`` — load into SGLang
      3. ``engine.generate(..., lora_path=[name])`` — generate with adapter
      4. Next trial: ``engine.unload_lora_adapter(name)`` → repeat from 1
    """

    def __init__(self, config: AbliterixConfig):
        import sglang as sgl

        self.config = config
        self._sgl = sgl

        tp = config.model.tensor_parallel_size
        if tp is None:
            tp = torch.cuda.device_count()

        model_id = config.model.model_id
        trust = config.model.trust_remote_code or False

        print(f"* Loading model in SGLang with TP={tp}...")

        kwargs: dict[str, Any] = dict(
            model_path=model_id,
            tp_size=tp,
            mem_fraction_static=config.model.gpu_memory_utilization,
            trust_remote_code=trust,
            # LoRA support — adapters loaded dynamically per trial.
            enable_lora=True,
            max_loras_per_batch=1,
            chunked_prefill_size=8192,
            # RadixAttention (prefix caching) is incompatible with LoRA.
            # Must be explicitly disabled.
            disable_radix_cache=True,
        )
        if config.model.max_model_len is not None:
            kwargs["context_length"] = config.model.max_model_len
        if config.model.max_num_seqs is not None:
            kwargs["max_running_requests"] = config.model.max_num_seqs

        # Model config overrides (e.g. MTP-3 → MTP-1 for Step-3.5-Flash).
        if config.model.hf_overrides:
            kwargs["hf_overrides"] = config.model.hf_overrides

        # FP8: only set explicitly if user specified quant_method="fp8".
        # Native FP8 models (config.json has quantization_config) are auto-detected.
        is_fp8 = config.model.quant_method and config.model.quant_method.value == "fp8"
        if is_fp8:
            kwargs["quantization"] = "fp8"

        # Also detect native FP8 models for KV cache logic.
        if not is_fp8:
            try:
                from transformers import AutoConfig

                _auto_cfg = AutoConfig.from_pretrained(
                    model_id, trust_remote_code=trust
                )
                _qcfg = getattr(_auto_cfg, "quantization_config", None)
                if _qcfg is None:
                    _text_cfg = getattr(_auto_cfg, "text_config", None)
                    if _text_cfg is not None:
                        _qcfg = getattr(_text_cfg, "quantization_config", None)
                if _qcfg is not None:
                    _qm = (
                        _qcfg
                        if isinstance(_qcfg, dict)
                        else getattr(_qcfg, "__dict__", {})
                    )
                    if _qm.get("quant_method") == "fp8":
                        is_fp8 = True
            except Exception:
                pass

        # FP8 KV cache on H100+.
        kv_dtype = config.model.kv_cache_dtype
        if kv_dtype is None and is_fp8:
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_capability(0)
                if cc[0] >= 9:
                    kv_dtype = "fp8_e4m3"
        if kv_dtype is not None:
            kwargs["kv_cache_dtype"] = kv_dtype

        self.engine = sgl.Engine(**kwargs)

        # Engine.get_tokenizer() does not exist — access via tokenizer_manager.
        self.tokenizer = self.engine.tokenizer_manager.tokenizer

        # Adapter management — use tmpfs for speed.
        tmpfs_base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        self._adapter_dir = os.path.join(
            tempfile.mkdtemp(prefix="abliterix_sglang_lora_", dir=tmpfs_base), "current"
        )
        self._adapter_loaded = False
        self._lora_target_modules: list[str] = []

        print(f"  [green]Ok[/] (SGLang TP={tp}, radix_cache=off for LoRA)")

    # ------------------------------------------------------------------
    # Chat template formatting
    # ------------------------------------------------------------------

    def _format_prompt(self, msg: ChatMessage) -> str:
        messages: list[dict[str, str]] = []
        if msg.system:
            messages.append({"role": "system", "content": msg.system})
        messages.append({"role": "user", "content": msg.user})
        kwargs: dict[str, Any] = dict(
            add_generation_prompt=True,
            tokenize=False,
        )
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _format_prompts(self, messages: list[ChatMessage]) -> list[str]:
        return [self._format_prompt(m) for m in messages]

    # ------------------------------------------------------------------
    # LoRA adapter serialisation and lifecycle
    # ------------------------------------------------------------------

    def save_adapter(
        self,
        lora_weights: dict[str, tuple[Tensor, Tensor]],
        target_modules: list[str],
        base_model_id: str,
    ) -> str:
        """Serialise LoRA weights to PEFT format and load into SGLang.

        Returns the adapter NAME (not path) for use in generate() calls.
        """
        adapter_dir = self._adapter_dir
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        os.makedirs(adapter_dir)

        state_dict: dict[str, Tensor] = {}
        for module_path, (lora_a, lora_b) in lora_weights.items():
            rank = lora_a.shape[0]
            if rank < _SGLANG_MIN_RANK:
                pad = _SGLANG_MIN_RANK - rank
                lora_a = F.pad(lora_a, (0, 0, 0, pad))
                lora_b = F.pad(lora_b, (0, pad, 0, 0))

            peft_key = f"base_model.model.{module_path}"
            state_dict[f"{peft_key}.lora_A.weight"] = lora_a.contiguous().cpu()
            state_dict[f"{peft_key}.lora_B.weight"] = lora_b.contiguous().cpu()

        save_file(state_dict, os.path.join(adapter_dir, "adapter_model.safetensors"))

        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": base_model_id,
            "r": _SGLANG_MIN_RANK,
            "lora_alpha": _SGLANG_MIN_RANK,
            "target_modules": target_modules,
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "inference_mode": True,
        }
        with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f)

        # SGLang LoRA lifecycle: unload previous → load new.
        # generate() takes the adapter NAME, not the filesystem path.
        if self._adapter_loaded:
            self.engine.unload_lora_adapter(_ADAPTER_NAME)
            self._adapter_loaded = False

        self.engine.load_lora_adapter(_ADAPTER_NAME, adapter_dir)
        self._adapter_loaded = True

        self._lora_target_modules = target_modules
        # Return the adapter NAME — this is what generate(lora_path=...) expects.
        return _ADAPTER_NAME

    # ------------------------------------------------------------------
    # Generation methods
    # ------------------------------------------------------------------

    def generate_text(
        self,
        messages: list[ChatMessage],
        skip_special_tokens: bool = False,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
        adapter_path: str | None = None,
    ) -> list[str]:
        prompts = self._format_prompts(messages)
        max_tok = max_new_tokens or self.config.inference.max_gen_tokens
        min_tok = min_new_tokens
        if min_tok is None and max_new_tokens is None:
            min_tok = self.config.inference.min_gen_tokens

        if min_tok is not None and min_tok > max_tok:
            raise ValueError(
                f"min_gen_tokens ({min_tok}) cannot exceed max_gen_tokens ({max_tok})"
            )

        sampling_params = {"temperature": 0, "max_new_tokens": max_tok}
        if min_tok is not None:
            sampling_params["min_new_tokens"] = min_tok

        # SGLang: lora_path is a separate kwarg to generate(), not part of
        # sampling_params.  It takes adapter NAMEs (not filesystem paths),
        # as a list matching the number of prompts.
        gen_kwargs: dict[str, Any] = {}
        if adapter_path:
            gen_kwargs["lora_path"] = [adapter_path] * len(prompts)

        outputs = self.engine.generate(prompts, sampling_params, **gen_kwargs)

        return [out["text"] for out in outputs]

    def generate_text_batched(
        self,
        messages: list[ChatMessage],
        skip_special_tokens: bool = False,
        max_new_tokens: int | None = None,
        min_new_tokens: int | None = None,
        adapter_path: str | None = None,
    ) -> list[str]:
        return self.generate_text(
            messages,
            skip_special_tokens=skip_special_tokens,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            adapter_path=adapter_path,
        )

    def generate_and_score(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int,
        kl_token_count: int,
        skip_special_tokens: bool = False,
        min_new_tokens: int | None = None,
        adapter_path: str | None = None,
    ) -> tuple[list[str], Tensor]:
        """Generate responses and retain every requested sampler distribution.

        The sampler distributions are useful for inspecting a single run.  For
        multi-token KL between two model states, use
        :meth:`score_continuation_logprobs_batched` so both states are scored on
        identical teacher-forced prefixes.
        """
        if kl_token_count < 1:
            raise ValueError("kl_token_count must be at least 1")
        if kl_token_count > max_new_tokens:
            raise ValueError(
                f"kl_token_count ({kl_token_count}) cannot exceed "
                f"max_new_tokens ({max_new_tokens})"
            )

        prompts = self._format_prompts(messages)

        k_logprobs = 100

        sampling_params: dict[str, Any] = {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        }
        requested_min_tokens = max(min_new_tokens or 0, kl_token_count)
        if requested_min_tokens > max_new_tokens:
            raise ValueError(
                f"min_gen_tokens ({requested_min_tokens}) cannot exceed "
                f"max_gen_tokens ({max_new_tokens})"
            )
        sampling_params["min_new_tokens"] = requested_min_tokens

        # SGLang: return_logprob, top_logprobs_num, lora_path are separate
        # keyword arguments to generate(), NOT part of sampling_params.
        gen_kwargs: dict[str, Any] = {
            "return_logprob": True,
            "top_logprobs_num": k_logprobs,
        }
        if adapter_path:
            gen_kwargs["lora_path"] = [adapter_path] * len(prompts)

        outputs = self.engine.generate(prompts, sampling_params, **gen_kwargs)
        if len(outputs) != len(messages):
            raise RuntimeError(
                "SGLang returned an unexpected number of generation results "
                f"({len(outputs)} != {len(messages)})"
            )

        responses: list[str] = []
        vocab_size = len(self.tokenizer)
        uniform_lp = math.log(1.0 / vocab_size)
        dense_logprobs = torch.full(
            (len(outputs), kl_token_count, vocab_size),
            -30.0,
            dtype=torch.float32,
        )
        has_finite = torch.zeros(
            (len(outputs), kl_token_count),
            dtype=torch.bool,
        )

        for batch_idx, out in enumerate(outputs):
            responses.append(out["text"])

            # SGLang meta_info.output_top_logprobs:
            #   list[list[tuple[logprob: float, token_id: int, text: str|None]]]
            # Outer list = per generated token position.
            # Inner list = top-K candidates at that position.
            meta = out.get("meta_info", {})
            top_lps = meta.get("output_top_logprobs", [])
            if not isinstance(top_lps, (list, tuple)):
                continue

            for step_idx, step_lps in enumerate(top_lps[:kl_token_count]):
                # Each item is (logprob, token_id, decoded_text_or_None).
                if isinstance(step_lps, (list, tuple)):
                    for item in step_lps:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            logprob_val = float(item[0])
                            token_id = int(item[1])
                            if (
                                math.isfinite(logprob_val)
                                and 0 <= token_id < vocab_size
                            ):
                                dense_logprobs[batch_idx, step_idx, token_id] = (
                                    logprob_val
                                )
                                has_finite[batch_idx, step_idx] = True

        dense_logprobs = F.log_softmax(dense_logprobs, dim=-1)
        dense_logprobs[~has_finite] = uniform_lp
        if kl_token_count == 1:
            dense_logprobs = dense_logprobs[:, 0, :]
        return responses, dense_logprobs

    def generate_and_score_batched(
        self,
        messages: list[ChatMessage],
        max_new_tokens: int,
        kl_token_count: int,
        skip_special_tokens: bool = False,
        min_new_tokens: int | None = None,
        adapter_path: str | None = None,
    ) -> tuple[list[str], Tensor]:
        return self.generate_and_score(
            messages,
            max_new_tokens=max_new_tokens,
            kl_token_count=kl_token_count,
            skip_special_tokens=skip_special_tokens,
            min_new_tokens=min_new_tokens,
            adapter_path=adapter_path,
        )

    def compute_logprobs_batched(
        self,
        messages: list[ChatMessage],
        adapter_path: str | None = None,
    ) -> Tensor:
        _, logprobs = self.generate_and_score(
            messages,
            max_new_tokens=self.config.kl.token_count,
            kl_token_count=self.config.kl.token_count,
            adapter_path=adapter_path,
        )
        return logprobs

    def score_continuation_logprobs_batched(
        self,
        messages: list[ChatMessage],
        continuations: list[str],
        token_count: int,
        adapter_path: str | None = None,
    ) -> Tensor:
        """Capture distributions on shared teacher-forced continuations.

        SGLang input logprobs are aligned to input token positions.  Requesting
        from the final prompt token produces one leading ``None`` alignment
        entry followed by distributions predicting continuation tokens 1..N.
        Requiring that marker prevents an API/version mismatch from silently
        falling back to incomparable free-running trajectories.

        Returns ``(batch, vocab)`` for one token and
        ``(batch, token_count, vocab)`` for multiple tokens.
        """
        if len(messages) != len(continuations):
            raise ValueError(
                "messages and continuations must have the same length "
                f"({len(messages)} != {len(continuations)})"
            )
        if token_count < 1:
            raise ValueError("token_count must be at least 1")

        prompts = self._format_prompts(messages)
        input_ids: list[list[int]] = []
        logprob_start_lens: list[int] = []

        for item_idx, (prompt, continuation) in enumerate(
            zip(prompts, continuations, strict=True)
        ):
            try:
                prompt_ids = list(
                    self.tokenizer.encode(prompt, add_special_tokens=False)
                )
                full_ids = list(
                    self.tokenizer.encode(
                        prompt + continuation,
                        add_special_tokens=False,
                    )
                )
            except TypeError:
                prompt_ids = list(self.tokenizer.encode(prompt))
                full_ids = list(self.tokenizer.encode(prompt + continuation))

            if not prompt_ids:
                raise ValueError(
                    f"Prompt item {item_idx} has no tokens; SGLang cannot align "
                    "the first continuation distribution"
                )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    f"Continuation item {item_idx} changes the prompt token boundary; "
                    "teacher-forced continuation positions are ambiguous"
                )

            stop = len(prompt_ids) + token_count
            if stop > len(full_ids):
                available = max(len(full_ids) - len(prompt_ids), 0)
                raise ValueError(
                    f"Continuation item {item_idx} has only {available} token(s); "
                    f"token_count={token_count} requires at least {token_count}"
                )

            # Only prefill the requested continuation prefix.  Starting one
            # token before it is required by SGLang's input-logprob alignment:
            # [None, P(c_0 | prompt), P(c_1 | prompt+c_0), ...].
            input_ids.append(full_ids[:stop])
            logprob_start_lens.append(len(prompt_ids) - 1)

        sampling_params: dict[str, Any] = {
            "temperature": 0,
            "max_new_tokens": 1,
        }
        gen_kwargs: dict[str, Any] = {
            "return_logprob": True,
            "logprob_start_len": logprob_start_lens,
            "top_logprobs_num": 100,
        }
        if adapter_path:
            gen_kwargs["lora_path"] = [adapter_path] * len(input_ids)

        outputs = self.engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            **gen_kwargs,
        )
        if len(outputs) != len(messages):
            raise RuntimeError(
                "SGLang returned an unexpected number of continuation scores "
                f"({len(outputs)} != {len(messages)})"
            )

        vocab_size = len(self.tokenizer)
        dense_logprobs = torch.full(
            (len(outputs), token_count, vocab_size),
            -30.0,
            dtype=torch.float32,
        )

        for batch_idx, out in enumerate(outputs):
            meta = out.get("meta_info", {})
            top_lps = meta.get("input_top_logprobs")
            if not isinstance(top_lps, (list, tuple)):
                raise RuntimeError(
                    "SGLang did not return input_top_logprobs required for "
                    f"teacher-forced continuation item {batch_idx}; refusing to "
                    "substitute free-running generated trajectories"
                )
            if len(top_lps) < token_count + 1:
                raise RuntimeError(
                    "SGLang returned too few input_top_logprobs positions for "
                    f"teacher-forced continuation item {batch_idx}: need "
                    f"{token_count + 1}, got {len(top_lps)}"
                )
            if top_lps[0] is not None:
                raise RuntimeError(
                    "SGLang input_top_logprobs did not contain the expected "
                    f"teacher-forced alignment marker for item {batch_idx}"
                )

            for step_idx, step_lps in enumerate(top_lps[1 : token_count + 1]):
                if not isinstance(step_lps, (list, tuple)) or not step_lps:
                    raise RuntimeError(
                        "SGLang returned no input_top_logprobs distribution for "
                        f"teacher-forced continuation item {batch_idx}, step {step_idx}"
                    )

                found_finite = False
                for item in step_lps:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    logprob_val = float(item[0])
                    token_id = int(item[1])
                    if not math.isfinite(logprob_val) or not 0 <= token_id < vocab_size:
                        continue
                    dense_logprobs[batch_idx, step_idx, token_id] = logprob_val
                    found_finite = True

                if not found_finite:
                    raise RuntimeError(
                        "SGLang returned no finite input_top_logprobs for "
                        f"teacher-forced continuation item {batch_idx}, step {step_idx}"
                    )

        dense_logprobs = F.log_softmax(dense_logprobs, dim=-1)
        if token_count == 1:
            return dense_logprobs[:, 0, :]
        return dense_logprobs

    def shutdown(self):
        """Shut down the SGLang engine and free GPU memory."""
        if hasattr(self, "engine") and self.engine is not None:
            if self._adapter_loaded:
                try:
                    self.engine.unload_lora_adapter(_ADAPTER_NAME)
                except Exception:
                    pass
            self.engine.shutdown()
            self.engine = None
