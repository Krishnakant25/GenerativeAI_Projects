"""Tests for multiple-comparisons correction.

We check the properties that matter for honest reporting rather than
re-deriving statsmodels' arithmetic: corrected p >= raw p, order preservation,
Bonferroni being at least as strict as FDR, and clean handling of edge cases.
"""

from __future__ import annotations

import pytest

from causal.correction import correct_p_values


def test_corrected_pvalues_never_below_raw():
    raw = [0.001, 0.01, 0.03, 0.2, 0.8]
    res = correct_p_values(raw, method="fdr_bh", alpha=0.05)
    assert len(res.corrected_p_values) == len(raw)
    for raw_p, corr_p in zip(raw, res.corrected_p_values):
        assert corr_p >= raw_p - 1e-12      # correction can only inflate p
        assert 0.0 <= corr_p <= 1.0


def test_order_is_preserved():
    raw = [0.5, 0.001, 0.2]
    res = correct_p_values(raw, method="fdr_bh")
    # The smallest raw p (index 1) must remain the smallest corrected p.
    assert res.corrected_p_values[1] == min(res.corrected_p_values)


def test_bonferroni_at_least_as_strict_as_fdr():
    raw = [0.001, 0.01, 0.02, 0.04, 0.5]
    fdr = correct_p_values(raw, method="fdr_bh", alpha=0.05)
    bonf = correct_p_values(raw, method="bonferroni", alpha=0.05)
    # Bonferroni rejects no more hypotheses than FDR.
    assert sum(bonf.reject) <= sum(fdr.reject)


def test_empty_input():
    res = correct_p_values([], method="fdr_bh")
    assert res.corrected_p_values == []
    assert res.reject == []


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        correct_p_values([0.01], method="not_a_method")
