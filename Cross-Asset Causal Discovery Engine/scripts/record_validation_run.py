"""Recorded, reproducible Phase-1 validation run -> results/ artifacts.

This is the single command that produces the project's "evidence pack": it runs
the full Layer-1 pipeline over a **fixed, closed date window** with **fixed
seeds**, stress-tests the strong results with a stationarity check, and writes
every output to disk so the README can be written around real numbers.

Reproducibility:
  * The window is pinned (``WINDOW_START`` / ``WINDOW_END``). Both ends are in
    the past, so yfinance returns stable historical data and re-running yields
    the same statistics (only the timestamped ``run_id`` differs).
  * The HMM seed is fixed in ``config.DEFAULT_HMM_RANDOM_STATE`` (asserted
    below). PC (causal-learn, fisherz) and Granger are deterministic given the
    data, so the candidate table, graph and regimes are reproducible.

Stress test (the project's whole point — statistical maturity over a flashy
number): before trusting an extreme Granger p-value, we run an Augmented
Dickey-Fuller test on every log-return series feeding the Granger stage and
record which are stationary. Granger assumes weakly-stationary inputs; a
non-stationary series would make an extreme p-value suspect. The finding is
recorded honestly either way.

Run from the project root:
    python scripts/record_validation_run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, no display needed
import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
from statsmodels.tsa.stattools import adfuller

# Make the project root importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from causal.pipeline import run_analysis
from config import ASSET_UNIVERSE, AssetClass, asset_name
from data.fetcher import fetch_panel
from data.preprocessor import preprocess
from db import storage

# ---------------------------------------------------------------------------
# Fixed, reproducible run configuration
# ---------------------------------------------------------------------------
WINDOW_START = "2018-01-03"
WINDOW_END = "2026-06-12"
ADF_ALPHA = 0.05  # ADF p < this => reject unit-root null => series is stationary

RESULTS = ROOT / "results"
SHOTS = RESULTS / "screenshots"

# Presentation colours, mirrored from the dashboard so artifacts read the same.
ASSET_CLASS_COLOR = {
    AssetClass.COMMODITY: "#E8A33D",
    AssetClass.CURRENCY: "#3FA66A",
    AssetClass.EQUITY_INDEX: "#3D7BE8",
    AssetClass.RATE: "#B05CED",
    AssetClass.SECTOR_ETF: "#E2585B",
}
EDGE_COLOR = {"directed": "#1f77b4", "undirected": "#7f7f7f", "bidirected": "#9467bd"}


def _fmt_p(p: float | None) -> str:
    return f"{p:.2e}" if p is not None else "n/a"


def _fmt_corr(c: float | None) -> str:
    return f"{c:+.3f}" if c is not None else "n/a"


# ===========================================================================
# Stationarity stress-test
# ===========================================================================

def run_stationarity(returns: pl.DataFrame) -> list[dict]:
    """ADF test on each log-return series feeding Granger. One row per ticker."""
    rows: list[dict] = []
    tickers = [c for c in returns.columns if c != "date"]
    for t in tickers:
        series = returns[t].to_numpy()
        stat, pval, used_lag, nobs, crit, _ = adfuller(series, autolag="AIC")
        cls = ASSET_UNIVERSE.get(t)
        rows.append(
            {
                "ticker": t,
                "name": asset_name(t),
                "asset_class": cls[1].value if cls else "unknown",
                "n_obs": int(nobs),
                "adf_stat": float(stat),
                "adf_pvalue": float(pval),
                "used_lag": int(used_lag),
                "crit_1pct": float(crit["1%"]),
                "crit_5pct": float(crit["5%"]),
                "crit_10pct": float(crit["10%"]),
                "stationary_5pct": bool(pval < ADF_ALPHA),
            }
        )
    rows.sort(key=lambda r: r["adf_pvalue"], reverse=True)  # least-stationary first
    return rows


# ===========================================================================
# Artifact tables
# ===========================================================================

def candidates_table(result) -> pl.DataFrame:
    meta = result.graph_meta
    rows = []
    for c in result.candidates:
        edge = meta.get(c.candidate_id)
        rows.append(
            {
                "candidate_id": c.candidate_id,
                "asset_a": c.asset_a,
                "asset_a_name": asset_name(c.asset_a),
                "asset_b": c.asset_b,
                "asset_b_name": asset_name(c.asset_b),
                "lag": c.lag,
                "granger_p_value": c.granger_p_value,
                "corrected_p_value": c.corrected_p_value,
                "correlation_strength": c.correlation_strength,
                "statistical_confidence": c.statistical_confidence,
                "is_significant": c.is_significant,
                "in_graph": edge is not None,
                "edge_type": edge[0] if edge else None,
                "orientation_source": edge[1] if edge else None,
            }
        )
    return pl.DataFrame(rows).sort("corrected_p_value")


def graph_edges_table(result) -> pl.DataFrame:
    by_id = {c.candidate_id: c for c in result.candidates}
    rows = []
    for cid, (edge_type, orientation) in result.graph_meta.items():
        c = by_id[cid]
        rows.append(
            {
                "source": c.asset_a,
                "source_name": asset_name(c.asset_a),
                "target": c.asset_b,
                "target_name": asset_name(c.asset_b),
                "edge_type": edge_type,
                "orientation_source": orientation,
                "corrected_p_value": c.corrected_p_value,
                "lag": c.lag,
                "correlation_strength": c.correlation_strength,
                "is_significant": c.is_significant,
            }
        )
    return pl.DataFrame(rows).sort("corrected_p_value")


def regimes_table(result) -> pl.DataFrame:
    rows = []
    for c in result.candidates:
        for p in c.regime_periods:
            rows.append(
                {
                    "asset_a": c.asset_a,
                    "asset_a_name": asset_name(c.asset_a),
                    "asset_b": c.asset_b,
                    "asset_b_name": asset_name(c.asset_b),
                    "start": p.start.isoformat(),
                    "end": p.end.isoformat(),
                    "active": p.active,
                    "mean_correlation": p.mean_correlation,
                }
            )
    return pl.DataFrame(rows)


# ===========================================================================
# Headless renders (no browser in this environment; these depict the real
# recorded numbers and are labelled as headless renders in run_summary.md).
# ===========================================================================

def render_graph_png(edges_df: pl.DataFrame, nodes: list[str], path: Path,
                     title: str, significant_only: bool = True) -> int:
    """Render the discovered causal graph with each edge labelled by its
    FDR-corrected Granger p-value (honesty rule: no bare arrows)."""
    df = edges_df
    if significant_only:
        df = df.filter(pl.col("is_significant"))

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for r in df.iter_rows(named=True):
        g.add_edge(r["source"], r["target"], **r)

    fig, ax = plt.subplots(figsize=(13, 9))
    pos = nx.spring_layout(g, seed=42, k=1.4, iterations=200)

    node_colors = [
        ASSET_CLASS_COLOR.get(
            ASSET_UNIVERSE[n][1] if n in ASSET_UNIVERSE else None, "#888888"
        )
        for n in g.nodes
    ]
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=1500,
                           edgecolors="#222", linewidths=1.0, ax=ax)
    nx.draw_networkx_labels(g, pos, labels={n: asset_name(n) for n in g.nodes},
                            font_size=8, font_color="#111", ax=ax)

    for r in df.iter_rows(named=True):
        style = "solid" if r["edge_type"] != "undirected" else "dashed"
        nx.draw_networkx_edges(
            g, pos, edgelist=[(r["source"], r["target"])],
            edge_color=EDGE_COLOR.get(r["edge_type"], "#7f7f7f"),
            style=style, width=1.6, arrowsize=16,
            arrows=r["edge_type"] != "undirected",
            connectionstyle="arc3,rad=0.08", ax=ax, node_size=1500,
        )
    edge_labels = {
        (r["source"], r["target"]): _fmt_p(r["corrected_p_value"])
        for r in df.iter_rows(named=True)
    }
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6,
                                 font_color="#1a1a1a", label_pos=0.5,
                                 bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                           ec="none", alpha=0.7), ax=ax)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.text(0.5, -0.04,
            "Edge label = FDR-corrected Granger p-value. Lag & correlation in "
            "graph_edges.csv. Granger = predictive precedence, NOT proof of "
            "causation.",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")
    # class legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                   markersize=10, label=cls.value.replace("_", " ").title())
        for cls, col in ASSET_CLASS_COLOR.items()
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, title="Asset class")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return g.number_of_edges()


def render_regime_png(regimes_df: pl.DataFrame, pair: tuple[str, str],
                      path: Path) -> None:
    a, b = pair
    sub = regimes_df.filter(
        (pl.col("asset_a") == a) & (pl.col("asset_b") == b)
    ).sort("start")
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    import datetime as _dt

    for r in sub.iter_rows(named=True):
        s = _dt.date.fromisoformat(r["start"])
        e = _dt.date.fromisoformat(r["end"])
        ax0.barh(0, (e - s).days, left=s, height=0.6,
                 color="#2E9E5B" if r["active"] else "#9AA0A6",
                 edgecolor="white")
        mc = r["mean_correlation"]
        ax1.bar(s, mc, width=(e - s).days, align="edge",
                color="#2E9E5B" if (mc or 0) >= 0 else "#E2585B",
                edgecolor="white", linewidth=0.3)
    ax0.set_yticks([])
    ax0.set_ylabel("Regime")
    ax0.set_title(f"Regime timeline — {asset_name(a)} -> {asset_name(b)}  "
                  f"(time-bound, not a permanent claim)",
                  fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch

    ax0.legend(handles=[Patch(color="#2E9E5B", label="Active (coupled)"),
                        Patch(color="#9AA0A6", label="Inactive (decoupled)")],
               loc="upper right", fontsize=8, ncol=2)
    ax1.axhline(0, color="#333", linewidth=0.6)
    ax1.set_ylabel("Mean lead-lag corr")
    ax1.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_edge_detail_png(edges_df: pl.DataFrame, regimes_df: pl.DataFrame,
                           path: Path) -> dict:
    top = edges_df.filter(pl.col("is_significant")).sort("corrected_p_value").row(
        0, named=True
    )
    a, b = top["source"], top["target"]
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 2]}
    )
    ax0.axis("off")
    ax0.set_title("Edge detail panel  (Explore -> By relationship)",
                  fontsize=12, fontweight="bold", loc="left")
    lines = [
        f"{asset_name(a)}  ->  {asset_name(b)}",
        "",
        f"Corrected p-value : {_fmt_p(top['corrected_p_value'])}",
        f"Lag               : {top['lag']} trading days",
        f"Correlation       : {_fmt_corr(top['correlation_strength'])}",
        f"Edge type         : {top['edge_type']}  (oriented by "
        f"{top['orientation_source']})",
        "",
        "Direction is predictive precedence (Granger), NOT proof of causation.",
        "[Phase 2] LLM economic-mechanism / plausibility card: not yet built.",
    ]
    ax0.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11,
             family="monospace", transform=ax0.transAxes)

    sub = regimes_df.filter(
        (pl.col("asset_a") == a) & (pl.col("asset_b") == b)
    ).sort("start")
    import datetime as _dt

    for r in sub.iter_rows(named=True):
        s = _dt.date.fromisoformat(r["start"])
        e = _dt.date.fromisoformat(r["end"])
        ax1.barh(0, (e - s).days, left=s, height=0.6,
                 color="#2E9E5B" if r["active"] else "#9AA0A6",
                 edgecolor="white")
    ax1.set_yticks([])
    ax1.set_title("Regime history (time-bound)", fontsize=10)
    ax1.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return top


def render_disclaimer_png(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor="#7a1f1f", alpha=0.12))
    msg = (
        "Read this first.  Granger causality measures PREDICTIVE PRECEDENCE, "
        "not proof of causation.\nEverything shown is a CANDIDATE HYPOTHESIS for "
        "human review, surfaced with its corrected\np-value — never a bare "
        "directional claim.  This is a research-screening tool, NOT a trading "
        "system."
    )
    ax.text(0.03, 0.5, msg, va="center", ha="left", fontsize=12,
            color="#7a1f1f", transform=ax.transAxes, wrap=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Summary
# ===========================================================================

def write_summary(result, stationarity: list[dict], edges_df: pl.DataFrame,
                  regimes_df: pl.DataFrame, n_graph_png_edges: int,
                  detail_pair: dict, path: Path) -> dict:
    run = result.run
    cands = result.candidates
    n_total = len(cands)
    n_sig = sum(1 for c in cands if c.is_significant)
    top5 = (
        edges_df.sort("corrected_p_value").head(5)
        if edges_df.height
        else edges_df
    )
    non_stationary = [r for r in stationarity if not r["stationary_5pct"]]
    worst = max(stationarity, key=lambda r: r["adf_pvalue"])

    top5_md = "\n".join(
        f"| {asset_name(r['source'])} → {asset_name(r['target'])} "
        f"| {_fmt_p(r['corrected_p_value'])} | {r['lag']} "
        f"| {_fmt_corr(r['correlation_strength'])} "
        f"| {r['edge_type']}/{r['orientation_source']} |"
        for r in top5.iter_rows(named=True)
    )

    if non_stationary:
        stationarity_md = (
            f"**{len(non_stationary)} of {len(stationarity)}** return series "
            f"are NON-stationary at the {int(ADF_ALPHA*100)}% ADF level "
            f"(p ≥ {ADF_ALPHA}): "
            + ", ".join(f"{r['ticker']} (p={r['adf_pvalue']:.3g})"
                        for r in non_stationary)
            + ". This is a documented caveat for any edge they feed."
        )
    else:
        stationarity_md = (
            f"**All {len(stationarity)} return series are stationary** at the "
            f"{int(ADF_ALPHA*100)}% ADF level (every ADF p < {ADF_ALPHA}; the "
            f"least-stationary is {worst['ticker']} at p={worst['adf_pvalue']:.3g}). "
            "The extreme Granger p-values are therefore NOT an artifact of "
            "non-stationary inputs — the strong results survive this check. "
            "Remaining caveats (best-lag selection, residual autocorrelation/"
            "heteroskedasticity, and multiple comparisons) are documented in the "
            "README."
        )

    missing = result.missing_tickers or []
    missing_md = ", ".join(missing) if missing else "none — all 13 tickers returned data"

    text = f"""# Recorded validation run — summary

**run_id:** `{run.run_id}`
**Recorded window (covered):** {run.start_date} → {run.end_date}
**Requested window:** {WINDOW_START} → {WINDOW_END}
**Correction:** {run.correction_method} · α = {run.alpha} · max_lag = {run.max_lag}
**HMM random_state (fixed):** {config.DEFAULT_HMM_RANDOM_STATE}

## Counts
- Assets analysed: **{len(run.asset_universe)}**
- Candidate relationships (ordered pairs): **{n_total}**
- Significant after correction: **{n_sig}**
- Edges kept in PC graph: **{edges_df.height}** ({int(edges_df.filter(pl.col('is_significant')).height)} of which are Granger-significant)
- Regime windows recorded: **{regimes_df.height}** across {regimes_df.select(['asset_a','asset_b']).unique().height if regimes_df.height else 0} pairs
- Tickers that failed to return data: {missing_md}

## Top 5 graph edges (most significant)
| Relationship | Corrected p | Lag (d) | Correlation | type/orientation |
|---|---|---|---|---|
{top5_md}

## Stationarity stress-test (ADF on every Granger-input series)
{stationarity_md}

Full per-series ADF results in `stationarity.csv`.

## Artifacts written to `results/`
- `candidates.csv` — all {n_total} candidates with corrected p, lag, correlation, edge_type, orientation_source, in_graph.
- `graph_edges.csv` — the {edges_df.height} discovered graph edges with statistics.
- `regimes.csv` — {regimes_df.height} time-bound regime windows.
- `stationarity.csv` — ADF result per return series.
- `graph.png` — headless render of the {n_graph_png_edges} significant graph edges (each labelled with its corrected p-value).
- `test_output.txt` — full pytest output (written separately by the test run).
- `screenshots/` — headless renders of the four dashboard views (graph, regime timeline, edge detail, disclaimer banner), generated from this recorded run because the sandbox has no browser. To capture live browser screenshots, run `streamlit run dashboard/app.py` and load `{run.run_id}`.

_Detail panel rendered for top edge: {asset_name(detail_pair['source'])} → {asset_name(detail_pair['target'])} (corrected p = {_fmt_p(detail_pair['corrected_p_value'])})._
"""
    path.write_text(text, encoding="utf-8")

    return {
        "run_id": run.run_id,
        "window": f"{run.start_date}..{run.end_date}",
        "assets": len(run.asset_universe),
        "candidates": n_total,
        "significant": n_sig,
        "graph_edges": edges_df.height,
        "graph_edges_significant": int(edges_df.filter(pl.col("is_significant")).height),
        "regime_windows": regimes_df.height,
        "missing": missing,
        "non_stationary": [r["ticker"] for r in non_stationary],
        "worst_adf": {"ticker": worst["ticker"], "adf_pvalue": worst["adf_pvalue"]},
        "top5": [
            {
                "edge": f"{r['source']}->{r['target']}",
                "corrected_p": r["corrected_p_value"],
                "lag": r["lag"],
                "corr": r["correlation_strength"],
            }
            for r in top5.iter_rows(named=True)
        ],
    }


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    assert config.DEFAULT_HMM_RANDOM_STATE == 0, (
        "HMM random_state must be fixed for reproducibility "
        f"(got {config.DEFAULT_HMM_RANDOM_STATE})."
    )
    RESULTS.mkdir(exist_ok=True)
    SHOTS.mkdir(exist_ok=True)
    storage.init_db(config.DB_PATH)

    print(f"[1/6] Fetching panel {WINDOW_START}..{WINDOW_END} for stationarity input")
    fetched = fetch_panel(start=WINDOW_START, end=WINDOW_END)
    _, returns = preprocess(fetched.prices)
    print(f"      aligned return rows: {returns.height}, "
          f"series: {[c for c in returns.columns if c != 'date']}")

    print("[2/6] ADF stationarity stress-test on each Granger-input series")
    stationarity = run_stationarity(returns)
    pl.DataFrame(stationarity).write_csv(RESULTS / "stationarity.csv")
    n_nonstat = sum(1 for r in stationarity if not r["stationary_5pct"])
    print(f"      non-stationary series @5%: {n_nonstat} / {len(stationarity)}")

    print("[3/6] Running full pipeline (Granger -> FDR -> PC -> HMM) — slow…")
    result = run_analysis(
        start_date=WINDOW_START, end_date=WINDOW_END,
        notes="recorded Phase-1 validation run",
    )
    storage.persist_run(result.run, result.candidates, result.graph_meta,
                        db_path=config.DB_PATH)
    print(f"      run_id: {result.run.run_id}")

    print("[4/6] Writing CSV artifacts")
    cands_df = candidates_table(result)
    edges_df = graph_edges_table(result)
    regimes_df = regimes_table(result)
    cands_df.write_csv(RESULTS / "candidates.csv")
    edges_df.write_csv(RESULTS / "graph_edges.csv")
    regimes_df.write_csv(RESULTS / "regimes.csv")

    print("[5/6] Rendering graph.png + dashboard view images (headless)")
    n_png = render_graph_png(
        edges_df, result.run.asset_universe, RESULTS / "graph.png",
        title="Discovered causal graph — significant edges "
              f"({result.run.start_date} → {result.run.end_date})",
    )
    render_graph_png(
        edges_df, result.run.asset_universe, SHOTS / "01_graph_view.png",
        title="Dashboard — Causal graph view",
    )
    # pick the pair with the most regime windows for the timeline render
    if regimes_df.height:
        counts = (regimes_df.group_by(["asset_a", "asset_b"])
                  .len().sort("len", descending=True))
        a, b = counts.row(0)[0], counts.row(0)[1]
        render_regime_png(regimes_df, (a, b), SHOTS / "02_regime_timeline.png")
    detail_top = render_edge_detail_png(edges_df, regimes_df,
                                        SHOTS / "03_edge_detail.png")
    render_disclaimer_png(SHOTS / "04_disclaimer_banner.png")

    print("[6/6] Writing run_summary.md")
    summary = write_summary(result, stationarity, edges_df, regimes_df,
                            n_png, detail_top, RESULTS / "run_summary.md")

    print("\n=== RECORDING COMPLETE ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
