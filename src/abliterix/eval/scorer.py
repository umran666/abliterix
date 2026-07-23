# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Trial scoring: damage, coherence, and multi-objective evaluation.

The :class:`TrialScorer` orchestrates baseline capture during init and then
provides :meth:`score_trial` to evaluate each Optuna trial.
"""

import statistics

import torch
import torch.nn.functional as F
from torch import Tensor

from ..data import load_prompt_dataset
from ..settings import AbliterixConfig
from ..types import ChatMessage
from ..util import print
from .detector import RefusalDetector
from .metrics import ComplianceResult, MetricResult


def _finite_logprobs(logprobs: Tensor) -> Tensor:
    """Return normalized finite log-probabilities for KL scoring."""
    cleaned = torch.nan_to_num(logprobs, nan=-30.0, posinf=0.0, neginf=-30.0)
    return F.log_softmax(cleaned, dim=-1, dtype=torch.float32)


def _safe_kl_divergence(current_logprobs: Tensor, baseline_logprobs: Tensor) -> float:
    """Compute mean per-distribution KL with normalized finite inputs.

    All leading dimensions are independent distributions.  This makes a
    ``(batch, step, vocab)`` input the arithmetic mean of per-token KL values,
    while preserving the historical behavior for ``(batch, vocab)`` inputs.
    """
    if current_logprobs.shape != baseline_logprobs.shape:
        raise ValueError(
            "Current and baseline log-probabilities must have the same shape: "
            f"got {tuple(current_logprobs.shape)} and "
            f"{tuple(baseline_logprobs.shape)}"
        )

    # Output offloading can place baseline and trial distributions on
    # different devices. Align before normalization without changing shape.
    if baseline_logprobs.device != current_logprobs.device:
        baseline_logprobs = baseline_logprobs.to(current_logprobs.device)

    current = _finite_logprobs(current_logprobs)
    baseline = _finite_logprobs(baseline_logprobs)
    per_distribution = F.kl_div(
        current,
        baseline,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    kl = per_distribution.mean()
    if torch.isfinite(kl):
        return max(0.0, float(kl.item()))
    return float("inf")


class TrialScorer:
    """Measures model damage, response coherence, and compliance.

    On construction the scorer records baseline logprobs, response lengths,
    and refusal counts against the un-steered model.  Each call to
    :meth:`score_trial` then returns a multi-objective tuple that Optuna
    minimises.
    """

    config: AbliterixConfig
    detector: RefusalDetector
    benign_msgs: list[ChatMessage]
    target_msgs: list[ChatMessage]
    baseline_logprobs: Tensor
    baseline_refusal_count: int
    baseline_mean_length: float
    baseline_stdev_length: float

    def __init__(
        self,
        config: AbliterixConfig,
        engine,
        detector: RefusalDetector,
        defer_baseline: bool = False,
    ):
        self.config = config
        self.detector = detector

        print()
        print(
            f"Loading benign evaluation prompts from [bold]{config.benign_eval_prompts.dataset}[/]..."
        )
        self.benign_msgs = load_prompt_dataset(config, config.benign_eval_prompts)
        print(f"* [bold]{len(self.benign_msgs)}[/] prompts loaded")

        print()
        print(
            f"Loading target evaluation prompts from [bold]{config.target_eval_prompts.dataset}[/]..."
        )
        self.target_msgs = load_prompt_dataset(config, config.target_eval_prompts)
        print(f"* [bold]{len(self.target_msgs)}[/] prompts loaded")

        if defer_baseline:
            # Baseline capture deferred until capture_baseline() is called
            # after the TP backend is loaded.  This avoids running expensive
            # generation on HF pipeline parallelism (~4 tok/s) before the
            # fast TP backend is available.
            self.baseline_logprobs = None
            self.baseline_mean_length = 1.0
            self.baseline_stdev_length = 1.0
            self.baseline_refusal_count = 0
            print("* [dim]Baseline capture deferred to TP backend phase[/]")
        else:
            self._capture_baseline(engine)

    def _capture_baseline(self, engine):
        """Capture baseline logprobs, response lengths, and refusal count.

        Automatically routes to the TP backend (vLLM/SGLang) if available,
        avoiding the slow HF pipeline-parallel generation path.
        """
        # Capture baseline logprobs and response lengths in a single pass.
        # Route to TP backend if available.
        print("* Obtaining probability distributions and baseline response lengths...")
        vllm_gen = getattr(engine, "_vllm_gen", None)
        if self._supports_vllm_continuation_kl(vllm_gen):
            base_responses = vllm_gen.generate_text_batched(
                self.benign_msgs,
                skip_special_tokens=True,
                max_new_tokens=self.config.inference.max_gen_tokens,
                min_new_tokens=self.config.inference.min_gen_tokens,
                adapter_path=None,
            )
            self.baseline_logprobs = None
            self.baseline_continuations = base_responses
            print(
                "* Scoring baseline continuations for "
                "vLLM fixed-continuation NLL drift..."
            )
            self.baseline_continuation_nll = vllm_gen.score_continuations_nll(
                self.benign_msgs,
                self.baseline_continuations,
                adapter_path=None,
            )
        elif vllm_gen is not None and self.config.kl.token_count > 1:
            base_responses = vllm_gen.generate_text_batched(
                self.benign_msgs,
                skip_special_tokens=True,
                max_new_tokens=self.config.inference.max_gen_tokens,
                min_new_tokens=max(
                    self.config.inference.min_gen_tokens or 0,
                    self.config.kl.token_count,
                ),
                adapter_path=None,
            )
            self.baseline_continuations = base_responses
            self.baseline_logprobs = vllm_gen.score_continuation_logprobs_batched(
                self.benign_msgs,
                self.baseline_continuations,
                self.config.kl.token_count,
                adapter_path=None,
            )
        elif vllm_gen is not None:
            base_responses, self.baseline_logprobs = (
                vllm_gen.generate_and_score_batched(
                    self.benign_msgs,
                    max_new_tokens=self.config.inference.max_gen_tokens,
                    kl_token_count=self.config.kl.token_count,
                    skip_special_tokens=True,
                    min_new_tokens=self.config.inference.min_gen_tokens,
                    adapter_path=None,
                )
            )
        elif self.config.kl.token_count > 1:
            # Multi-token KL must compare both models on the same contexts.
            # First generate one baseline continuation per prompt, then score
            # its teacher-forced prefixes under the unsteered model.
            base_responses = engine.generate_text_batched(
                self.benign_msgs,
                skip_special_tokens=True,
                max_new_tokens=self.config.inference.max_gen_tokens,
                min_new_tokens=max(
                    self.config.inference.min_gen_tokens or 0,
                    self.config.kl.token_count,
                ),
            )
            self.baseline_continuations = base_responses
            self.baseline_logprobs = engine.score_continuation_logprobs_batched(
                self.benign_msgs,
                self.baseline_continuations,
                self.config.kl.token_count,
            )
        else:
            base_responses, self.baseline_logprobs = engine.generate_and_score_batched(
                self.benign_msgs,
                max_new_tokens=self.config.inference.max_gen_tokens,
                kl_token_count=self.config.kl.token_count,
                skip_special_tokens=True,
                min_new_tokens=self.config.inference.min_gen_tokens,
            )
        if not hasattr(self, "baseline_continuations"):
            self.baseline_continuations = None
            self.baseline_continuation_nll = None
        base_lengths = [len(r.split()) for r in base_responses]
        self.baseline_mean_length = (
            statistics.mean(base_lengths) if base_lengths else 1.0
        )
        self.baseline_stdev_length = (
            statistics.stdev(base_lengths) if len(base_lengths) > 1 else 1.0
        )
        print(
            f"* Baseline response length: [bold]{self.baseline_mean_length:.1f}[/] "
            f"+/- {self.baseline_stdev_length:.1f} words"
        )

        print("* Counting model refusals...")
        evaluate_result = getattr(
            self.detector,
            "evaluate_compliance_result",
            None,
        )
        if callable(evaluate_result):
            baseline_compliance = evaluate_result(
                engine,
                self.target_msgs,
            )
            if baseline_compliance.known_count == 0 and self.target_msgs:
                raise RuntimeError(
                    "Baseline refusal evaluation produced no known labels; "
                    "cannot construct a compliance objective."
                )
            self.baseline_compliance_result = baseline_compliance
            self.baseline_refusal_count = baseline_compliance.refusal_count
            self.baseline_evaluated_count = baseline_compliance.known_count
            print(
                f"* Initial refusals: [bold]{self.baseline_refusal_count}[/]"
                f"/{self.baseline_evaluated_count} known "
                f"({baseline_compliance.unknown_count} unknown)"
            )
        else:
            # Compatibility with detector-like integrations that predate the
            # per-sample result contract. Such evaluators cannot report
            # unknowns, so retain their historical all-samples denominator.
            self.baseline_refusal_count = self.detector.evaluate_compliance(
                engine,
                self.target_msgs,
            )
            self.baseline_evaluated_count = len(self.target_msgs)
            print(
                f"* Initial refusals: [bold]{self.baseline_refusal_count}[/]"
                f"/{self.baseline_evaluated_count}"
            )

    # ------------------------------------------------------------------
    # Individual metric helpers
    # ------------------------------------------------------------------

    def measure_damage(self, engine) -> MetricResult:
        """Measure model damage while preserving the estimator's identity."""
        print("  * Obtaining probability distributions...")
        vllm_gen = getattr(engine, "_vllm_gen", None)
        if self._use_vllm_continuation_kl(vllm_gen):
            adapter_path = getattr(engine, "_current_adapter_path", None)
            result = MetricResult(
                name="continuation_nll_drift",
                estimator="fixed_continuation_mean_absolute_error",
                units="nats/token",
                value=self._measure_vllm_continuation_kl(
                    vllm_gen,
                    adapter_path,
                ),
            )
        else:
            result = MetricResult(
                name="full_distribution_kl",
                estimator="mean_per_token_kl",
                units="nats/token",
                value=self._measure_full_distribution_kl(engine),
            )
        self.last_damage_metric = result
        print(
            f"  * {result.name} ({result.estimator}): "
            f"[bold]{result.value:.4f}[/] {result.units}"
        )
        return result

    def measure_kl_divergence(self, engine) -> float:
        """Return the legacy numeric damage value.

        New callers should use :meth:`measure_damage`, because vLLM in-place
        editing uses fixed-continuation NLL drift instead of KL.
        """
        return self.measure_damage(engine).value

    def _measure_full_distribution_kl(self, engine) -> float:
        """Compute full-distribution KL for backends with valid logprobs."""
        vllm_gen = getattr(engine, "_vllm_gen", None)
        adapter_path = getattr(engine, "_current_adapter_path", None)
        if vllm_gen is not None and self.config.kl.token_count > 1:
            if self.baseline_continuations is None:
                raise RuntimeError(
                    "Multi-token TP KL requires captured baseline continuations"
                )
            logprobs = vllm_gen.score_continuation_logprobs_batched(
                self.benign_msgs,
                self.baseline_continuations,
                self.config.kl.token_count,
                adapter_path=adapter_path,
            )
        elif vllm_gen is not None:
            logprobs = vllm_gen.compute_logprobs_batched(
                self.benign_msgs,
                adapter_path=adapter_path,
            )
        elif self.config.kl.token_count > 1:
            if self.baseline_continuations is None:
                raise RuntimeError(
                    "Multi-token HF KL requires captured baseline continuations"
                )
            logprobs = engine.score_continuation_logprobs_batched(
                self.benign_msgs,
                self.baseline_continuations,
                self.config.kl.token_count,
            )
        else:
            logprobs = engine.compute_logprobs_batched(self.benign_msgs)
        return _safe_kl_divergence(logprobs, self.baseline_logprobs)

    def measure_coherence(self, engine) -> float:
        """Compute how much steered response lengths deviate from baseline.

        Returns the mean absolute z-score of word counts relative to the
        un-steered model.  Values near 0 indicate unchanged fluency; values
        above 2 suggest degenerate repetition or truncation.
        """
        vllm_gen = getattr(engine, "_vllm_gen", None)
        adapter_path = getattr(engine, "_current_adapter_path", None)
        if vllm_gen is not None:
            responses = vllm_gen.generate_text_batched(
                self.benign_msgs,
                skip_special_tokens=True,
                max_new_tokens=self.config.inference.max_gen_tokens,
                min_new_tokens=self.config.inference.min_gen_tokens,
                adapter_path=adapter_path,
            )
        else:
            responses = engine.generate_text_batched(
                self.benign_msgs,
                skip_special_tokens=True,
                max_new_tokens=self.config.inference.max_gen_tokens,
                min_new_tokens=self.config.inference.min_gen_tokens,
            )
        lengths = [len(r.split()) for r in responses]
        if not lengths or self.baseline_stdev_length == 0:
            return 0.0
        current_mean = statistics.mean(lengths)
        return abs(current_mean - self.baseline_mean_length) / max(
            self.baseline_stdev_length,
            1.0,
        )

    def measure_damage_and_coherence(
        self,
        engine,
    ) -> tuple[MetricResult, float]:
        """Compute an identified damage metric and coherence in shared work.

        The single-token path uses one generation pass; the multi-token path
        additionally scores a fixed baseline continuation so both
        distributions are conditioned on identical prefixes.
        """
        print("  * Obtaining probability distributions and response lengths...")

        vllm_gen = getattr(engine, "_vllm_gen", None)
        adapter_path = getattr(engine, "_current_adapter_path", None)

        use_continuation_nll_drift = self._use_vllm_continuation_kl(vllm_gen)
        if use_continuation_nll_drift:
            responses = vllm_gen.generate_text_batched(
                self.benign_msgs,
                skip_special_tokens=True,
                max_new_tokens=self.config.inference.max_gen_tokens,
                min_new_tokens=self.config.inference.min_gen_tokens,
                adapter_path=adapter_path,
            )
            damage_value = self._measure_vllm_continuation_kl(
                vllm_gen,
                adapter_path,
            )
        elif vllm_gen is not None and self.config.kl.token_count > 1:
            if self.baseline_continuations is None:
                raise RuntimeError(
                    "Multi-token TP KL requires captured baseline continuations"
                )
            responses = vllm_gen.generate_text_batched(
                self.benign_msgs,
                skip_special_tokens=True,
                max_new_tokens=self.config.inference.max_gen_tokens,
                min_new_tokens=self.config.inference.min_gen_tokens,
                adapter_path=adapter_path,
            )
            logprobs = vllm_gen.score_continuation_logprobs_batched(
                self.benign_msgs,
                self.baseline_continuations,
                self.config.kl.token_count,
                adapter_path=adapter_path,
            )
            damage_value = _safe_kl_divergence(
                logprobs,
                self.baseline_logprobs,
            )
        elif vllm_gen is not None:
            responses, logprobs = vllm_gen.generate_and_score_batched(
                self.benign_msgs,
                max_new_tokens=self.config.inference.max_gen_tokens,
                kl_token_count=self.config.kl.token_count,
                skip_special_tokens=True,
                min_new_tokens=self.config.inference.min_gen_tokens,
                adapter_path=adapter_path,
            )
            damage_value = _safe_kl_divergence(
                logprobs,
                self.baseline_logprobs,
            )
        elif self.config.kl.token_count > 1:
            if self.baseline_continuations is None:
                raise RuntimeError(
                    "Multi-token HF KL requires captured baseline continuations"
                )
            responses = engine.generate_text_batched(
                self.benign_msgs,
                skip_special_tokens=True,
                max_new_tokens=self.config.inference.max_gen_tokens,
                min_new_tokens=self.config.inference.min_gen_tokens,
            )
            logprobs = engine.score_continuation_logprobs_batched(
                self.benign_msgs,
                self.baseline_continuations,
                self.config.kl.token_count,
            )
            damage_value = _safe_kl_divergence(
                logprobs,
                self.baseline_logprobs,
            )
        else:
            responses, logprobs = engine.generate_and_score_batched(
                self.benign_msgs,
                max_new_tokens=self.config.inference.max_gen_tokens,
                kl_token_count=self.config.kl.token_count,
                skip_special_tokens=True,
                min_new_tokens=self.config.inference.min_gen_tokens,
            )
            damage_value = _safe_kl_divergence(
                logprobs,
                self.baseline_logprobs,
            )

        if use_continuation_nll_drift:
            damage_metric = MetricResult(
                name="continuation_nll_drift",
                estimator="fixed_continuation_mean_absolute_error",
                units="nats/token",
                value=damage_value,
            )
        else:
            damage_metric = MetricResult(
                name="full_distribution_kl",
                estimator="mean_per_token_kl",
                units="nats/token",
                value=damage_value,
            )
        self.last_damage_metric = damage_metric
        print(
            f"  * {damage_metric.name} ({damage_metric.estimator}): "
            f"[bold]{damage_metric.value:.4f}[/] {damage_metric.units}"
        )

        lengths = [len(r.split()) for r in responses]
        if not lengths or self.baseline_stdev_length == 0:
            deviation = 0.0
        else:
            current_mean = statistics.mean(lengths)
            deviation = abs(current_mean - self.baseline_mean_length) / max(
                self.baseline_stdev_length,
                1.0,
            )
        print(f"  * Response length deviation: [bold]{deviation:.2f}[/] std devs")

        return damage_metric, deviation

    def measure_kl_and_coherence(self, engine) -> tuple[float, float]:
        """Return the legacy numeric damage/coherence pair.

        The first value may be either full-distribution KL or fixed-
        continuation NLL drift.  New callers should use
        :meth:`measure_damage_and_coherence` to preserve that identity.
        """
        damage_metric, deviation = self.measure_damage_and_coherence(engine)
        return damage_metric.value, deviation

    def _use_vllm_continuation_kl(self, vllm_gen) -> bool:
        """Use fixed-continuation NLL drift for vLLM in-place edits.

        Sparse top-k sampler KL is known to read as exactly zero on Gemma 4
        vLLM in-place runs even when refusal counts move.  This path keeps the
        ordinary KL estimator for HF/LoRA/SGLang and only swaps the metric for
        the edit mode affected by stale sampler logprobs.
        """
        return bool(
            self._supports_vllm_continuation_kl(vllm_gen)
            and getattr(self, "baseline_continuations", None) is not None
            and getattr(self, "baseline_continuation_nll", None) is not None
        )

    def _supports_vllm_continuation_kl(self, vllm_gen) -> bool:
        return bool(
            vllm_gen is not None
            and getattr(self.config.model, "use_in_place_editing", False)
            and hasattr(vllm_gen, "score_continuations_nll")
        )

    def _measure_vllm_continuation_kl(
        self, vllm_gen, adapter_path: str | None
    ) -> float:
        current_nll = vllm_gen.score_continuations_nll(
            self.benign_msgs,
            self.baseline_continuations,
            adapter_path=adapter_path,
        )
        baseline_nll = self.baseline_continuation_nll.to(current_nll.device)
        drift = torch.mean(torch.abs(current_nll - baseline_nll)).item()
        if torch.isfinite(torch.tensor(drift)):
            return float(drift)
        return float("inf")

    # ------------------------------------------------------------------
    # Multi-objective scoring
    # ------------------------------------------------------------------

    def _compute_objectives(
        self,
        kl_divergence: float,
        detected_refusals: int,
        length_deviation: float = 0.0,
        *,
        evaluated_count: int | None = None,
        baseline_refusals: int | None = None,
        baseline_evaluated_count: int | None = None,
        compliance_objective_override: float | None = None,
    ) -> tuple[float, float]:
        """Build the damage/compliance objective pair.

        Compliance is normally relative to the baseline refusal count.  A
        zero-refusal baseline has no meaningful ratio, so that edge case uses
        the current absolute refusal rate.  Explicit evaluated counts allow
        unknown judge labels to be excluded from both denominators.
        """
        scale = self.config.kl.scale
        if compliance_objective_override is not None:
            compliance_objective = compliance_objective_override
        else:
            baseline_refusals = (
                self.baseline_refusal_count
                if baseline_refusals is None
                else baseline_refusals
            )
            target_count = len(getattr(self, "target_msgs", ()))
            baseline_evaluated_count = (
                getattr(self, "baseline_evaluated_count", None)
                if baseline_evaluated_count is None
                else baseline_evaluated_count
            )
            if baseline_evaluated_count is None:
                baseline_evaluated_count = (
                    target_count or self.baseline_refusal_count or 1
                )
            if evaluated_count is None:
                evaluated_count = target_count or baseline_evaluated_count
            if evaluated_count < 1 or baseline_evaluated_count < 1:
                raise ValueError(
                    "Compliance objectives require at least one known label"
                )

            current_rate = detected_refusals / evaluated_count
            baseline_rate = baseline_refusals / baseline_evaluated_count
            compliance_objective = (
                current_rate / baseline_rate if baseline_rate > 0 else current_rate
            )

        # Default ("independent"): keep damage and compliance as independent
        # objectives so the optimizer can learn a real Pareto frontier.
        # Opt-in ("do_nothing_guard") retains Heretic's conservative-region
        # escape behavior for users who explicitly request it.
        if (
            self.config.kl.objective_mode == "do_nothing_guard"
            and kl_divergence < self.config.kl.target
        ):
            divergence_objective = compliance_objective * self.config.kl.target / scale
        else:
            divergence_objective = kl_divergence / scale

        # Penalise degenerate output lengths beyond 2 standard deviations.
        # Early-token distribution metrics cannot see long-form drift (model
        # collapses into "帮好帮好…" loops 50 tokens in but the prefix
        # logprobs look fine).  Length deviation catches both shrinkage and
        # bloating.  Threshold at 2σ keeps natural variation uncounted;
        # beyond that, multiply divergence by (1 + 0.1·(dev - 2)) so the
        # penalty scales with the damage metric rather than swamping
        # it (which an additive penalty did during v3 — it pushed damage
        # 2000× below target by accident).
        if length_deviation > 2.0:
            divergence_objective *= 1.0 + 0.1 * (length_deviation - 2.0)

        return (divergence_objective, compliance_objective)

    def measure_compliance_objective(self, engine) -> tuple[int, float]:
        """Measure refusals and build an unknown-safe compliance objective.

        When per-sample labels are available, baseline and trial rates are
        compared only on their shared known subset.  This prevents a failed
        judge batch from changing either the numerator or denominator.
        """
        print("  * Counting model refusals...")
        evaluate_result = getattr(
            self.detector,
            "evaluate_compliance_result",
            None,
        )
        compliance = (
            evaluate_result(engine, self.target_msgs)
            if callable(evaluate_result)
            else None
        )
        if compliance is None:
            detected_refusals = self.detector.evaluate_compliance(
                engine,
                self.target_msgs,
            )
            print(f"  * Refusals: [bold]{detected_refusals}[/]/{len(self.target_msgs)}")
            compliance_objective = self._compute_objectives(
                0.0,
                detected_refusals,
            )[1]
            return detected_refusals, compliance_objective

        self.last_compliance_result = compliance
        detected_refusals = compliance.refusal_count
        print(
            f"  * Refusals: [bold]{detected_refusals}[/]"
            f"/{compliance.known_count} known "
            f"({compliance.unknown_count} unknown)"
        )

        baseline = getattr(self, "baseline_compliance_result", None)
        if isinstance(baseline, ComplianceResult):
            if (
                baseline.evaluator != compliance.evaluator
                or baseline.protocol_version != compliance.protocol_version
            ):
                raise RuntimeError(
                    "Baseline and trial refusal evaluations used different "
                    "evaluator identities; refusing to compare them."
                )
            if len(baseline.labels) != len(compliance.labels):
                raise RuntimeError(
                    "Baseline and trial refusal evaluations returned different "
                    "sample counts; refusing to compare them."
                )
            paired = [
                (base, current)
                for base, current in zip(baseline.labels, compliance.labels)
                if base is not None and current is not None
            ]
            if not paired:
                raise RuntimeError(
                    "Baseline and trial refusal evaluations have no shared "
                    "known labels; refusing to produce a compliance metric."
                )
            paired_baseline_refusals = sum(base is True for base, _ in paired)
            paired_current_refusals = sum(current is True for _, current in paired)
            paired_count = len(paired)
        else:
            if compliance.known_count == 0:
                raise RuntimeError(
                    "Trial refusal evaluation produced no known labels; "
                    "refusing to produce a compliance metric."
                )
            paired_baseline_refusals = self.baseline_refusal_count
            paired_current_refusals = detected_refusals
            paired_count = compliance.known_count

        compliance_objective = self._compute_objectives(
            0.0,
            paired_current_refusals,
            evaluated_count=paired_count,
            baseline_refusals=paired_baseline_refusals,
            baseline_evaluated_count=paired_count,
        )[1]
        return detected_refusals, compliance_objective

    def score_trial(self, engine) -> tuple[tuple[float, float], float, int, float]:
        """Evaluate the current steered model and return the multi-objective score.

        Returns
        -------
        objectives : tuple[float, float]
            ``(divergence_objective, compliance_objective)`` to minimise.
        kl_divergence : float
            Legacy numeric damage slot. Its estimator identity is available
            as :attr:`last_damage_metric`.
        detected_refusals : int
            Number of target prompts classified as refusals.
        length_deviation : float
            Response-length z-score relative to the baseline.
        """
        damage_value, length_deviation = self.measure_kl_and_coherence(engine)
        detected_refusals, compliance_objective = self.measure_compliance_objective(
            engine
        )
        objectives = self._compute_objectives(
            damage_value,
            detected_refusals,
            length_deviation,
            compliance_objective_override=compliance_objective,
        )

        return objectives, damage_value, detected_refusals, length_deviation
