"""Multiple-comparisons correction.

Testing 13 assets pairwise in both directions is 13 x 12 = 156 simultaneous
Granger tests. At alpha = 0.05 you would expect ~8 "significant" results by
pure chance even if no real relationship existed anywhere. Reporting raw
p-values here would be statistically dishonest — so every run applies a
correction and downstream code reports the **corrected** p-value alongside
each edge (a project hard rule).

Two methods are supported:
  - ``fdr_bh``     Benjamini-Hochberg, controls the False Discovery Rate.
                   The sensible default for screening/exploratory work: it
                   tolerates a controlled fraction of false positives in
                   exchange for far more power than Bonferroni.
  - ``bonferroni`` Controls the Family-Wise Error Rate. Very conservative;
                   appropriate if even one false positive is costly.

Thin wrapper over ``statsmodels.stats.multitest.multipletests`` so the rest of
the codebase has one obvious entry point and a stable return shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from statsmodels.stats.multitest import multipletests

_METHOD_MAP = {
    "fdr_bh": "fdr_bh",
    "bonferroni": "bonferroni",
}


@dataclass(frozen=True)
class CorrectionResult:
    """Aligned, order-preserving output of a correction over many p-values."""

    corrected_p_values: list[float]
    reject: list[bool]      # True where the null is rejected (significant)
    method: str
    alpha: float


def correct_p_values(
    p_values: list[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> CorrectionResult:
    """Apply a multiple-comparisons correction to a list of raw p-values.

    Order is preserved: ``result.corrected_p_values[i]`` corresponds to
    ``p_values[i]``. An empty input returns an empty result rather than raising.
    """
    if method not in _METHOD_MAP:
        raise ValueError(
            f"Unknown correction method {method!r}; "
            f"expected one of {sorted(_METHOD_MAP)}"
        )
    if not p_values:
        return CorrectionResult([], [], method, alpha)

    reject, corrected, _, _ = multipletests(
        p_values, alpha=alpha, method=_METHOD_MAP[method]
    )
    return CorrectionResult(
        corrected_p_values=[float(p) for p in corrected],
        reject=[bool(r) for r in reject],
        method=method,
        alpha=alpha,
    )
