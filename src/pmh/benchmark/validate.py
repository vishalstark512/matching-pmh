"""Pass/fail checks on falsification arm reports (CI-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pmh.benchmark.protocol import BenchmarkResult


@dataclass
class ValidationReport:
    """Result of :func:`validate_falsification`."""

    passed: bool
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"PMH falsification validation: {status}"]
        for c in self.checks:
            lines.append(f"  ok: {c}")
        for f in self.failures:
            lines.append(f"  FAIL: {f}")
        return "\n".join(lines)


def validate_falsification(
    result: BenchmarkResult | dict[str, Any],
    *,
    min_margin: float = 0.0,
    require_arms: tuple[str, ...] = ("matched", "wrong_w", "isotropic"),
) -> ValidationReport:
    """Check matched beats wrong-W and isotropic on target metric (Lemma C spirit)."""
    if isinstance(result, BenchmarkResult):
        arms = result.arms
    else:
        arms = {
            k: type("_Row", (), {"val_metric": v.get("val_metric")})()
            for k, v in result.get("arms", {}).items()
        }

    failures: list[str] = []
    checks: list[str] = []

    missing = [a for a in require_arms if a not in arms]
    if missing:
        failures.append(f"missing arms: {missing}")
        return ValidationReport(passed=False, checks=checks, failures=failures)

    m = arms["matched"].val_metric
    w = arms["wrong_w"].val_metric
    i = arms["isotropic"].val_metric
    if m is None or w is None or i is None:
        failures.append("matched/wrong_w/isotropic missing val_metric")
        return ValidationReport(passed=False, checks=checks, failures=failures)

    if m <= w + min_margin:
        failures.append(f"matched ({m:.4f}) should beat wrong_w ({w:.4f})")
    else:
        checks.append(f"matched ({m:.4f}) > wrong_w ({w:.4f})")

    if m <= i + min_margin:
        failures.append(f"matched ({m:.4f}) should beat isotropic ({i:.4f})")
    else:
        checks.append(f"matched ({m:.4f}) > isotropic ({i:.4f})")

    return ValidationReport(passed=not failures, checks=checks, failures=failures)
