# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Public metric value objects used by evaluation protocols."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MetricResult:
    """A numeric metric whose identity and units travel with its value."""

    name: str
    estimator: str
    units: str
    value: float


@dataclass(frozen=True, slots=True)
class ComplianceResult:
    """Per-sample refusal labels with evaluator provenance.

    ``None`` is an unknown label caused by evaluator transport or parsing
    failure.  Unknown labels are deliberately distinct from compliance and
    refusal so callers can exclude them from metric denominators.
    """

    labels: tuple[bool | None, ...]
    evaluator: str
    protocol_version: str
    issues: tuple[str, ...] = ()

    @property
    def refusal_count(self) -> int:
        return sum(label is True for label in self.labels)

    @property
    def compliance_count(self) -> int:
        return sum(label is False for label in self.labels)

    @property
    def known_count(self) -> int:
        return self.refusal_count + self.compliance_count

    @property
    def unknown_count(self) -> int:
        return len(self.labels) - self.known_count

    @property
    def refusal_rate(self) -> float | None:
        if self.known_count == 0:
            return None
        return self.refusal_count / self.known_count

    def require_complete(self) -> "ComplianceResult":
        """Return self or reject a legacy metric that would hide unknowns."""
        if self.unknown_count:
            raise RuntimeError(
                "Refusal evaluation has "
                f"{self.unknown_count}/{len(self.labels)} unknown labels; "
                "refusing to report them as model behavior."
            )
        return self
