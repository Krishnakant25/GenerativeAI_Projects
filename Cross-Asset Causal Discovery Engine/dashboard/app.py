"""Streamlit dashboard for the Cross-Asset Causal Discovery Engine — Phase 1.

A **thin client** over the FastAPI service. It renders the statistical findings
(causal graph, regime timelines, per-edge evidence) served by ``api/main.py``
and never re-runs or imports the pipeline — the API is the single source of
truth (see ``dashboard/api_client.py``).

Honesty framing is a project hard rule, enforced in the UI as well:
  * a standing disclaimer banner (Granger = predictive precedence, not proof;
    these are candidate hypotheses for human review; not a trading system);
  * no causal arrow is ever drawn without its corrected p-value, lag and
    correlation attached.

Phase 2 (the LLM "hypothesis card" / plausibility layer) does NOT exist yet and
appears only as a clearly-labelled placeholder — never faked.

Run it (two shells):
    uvicorn api.main:app                 # 1) start the API
    streamlit run dashboard/app.py       # 2) start the dashboard
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import polars as pl
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

# Make the project root importable so ``import config`` works no matter the CWD
# Streamlit was launched from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402  (must follow the sys.path shim)
from config import ASSET_UNIVERSE, AssetClass, asset_name  # noqa: E402
from dashboard.api_client import APIError, CausalAPIClient  # noqa: E402

# ===========================================================================
# Presentation constants (pure styling — kept out of config.py, which holds
# behavioural parameters only).
# ===========================================================================

ASSET_CLASS_COLOR: dict[AssetClass, str] = {
    AssetClass.COMMODITY: "#E8A33D",      # amber
    AssetClass.CURRENCY: "#3FA66A",       # green
    AssetClass.EQUITY_INDEX: "#3D7BE8",   # blue
    AssetClass.RATE: "#B05CED",           # purple
    AssetClass.SECTOR_ETF: "#E2585B",     # red
}

# Edge encoding by CPDAG edge_type. orientation_source (pc/granger) is surfaced
# in every edge's hover tooltip and in the legend.
EDGE_STYLE: dict[str, dict] = {
    "directed":   {"color": "#4F9BFF", "dashes": False, "arrows": "to"},
    "undirected": {"color": "#9AA0A6", "dashes": True,  "arrows": ""},
    "bidirected": {"color": "#C792EA", "dashes": False, "arrows": "to, from"},
}

ACTIVE_COLOR = "#2E9E5B"
INACTIVE_COLOR = "#9AA0A6"
POS_CORR_COLOR = "#2E9E5B"
NEG_CORR_COLOR = "#E2585B"

# Layer-2 plausibility-flag presentation. The flag is a HEURISTIC FILTER, not
# validation — the colour encodes the LLM's judgement, never statistical truth.
FLAG_STYLE: dict[str, dict] = {
    "plausible_known_mechanism": {
        "label": "Plausible — known mechanism",
        "color": "#2E9E5B", "emoji": "🟢",
    },
    "plausible_novel": {
        "label": "Plausible — novel",
        "color": "#3D7BE8", "emoji": "🔵",
    },
    "likely_spurious": {
        "label": "Likely spurious",
        "color": "#E2585B", "emoji": "🔴",
    },
    "parse_failed": {
        "label": "Parse failed",
        "color": "#9AA0A6", "emoji": "⚪",
    },
}
# Stable display / filter order.
FLAG_ORDER = [
    "plausible_known_mechanism",
    "plausible_novel",
    "likely_spurious",
    "parse_failed",
]

# Phase-3 regime-flip lifecycle presentation. The badge encodes TRUST EARNED,
# not statistical truth: a 'confirmed' flip survived the confirmation window, a
# 'pending' flip is explicitly provisional (may still revert), a 'reverted' flip
# was rejected. Styling is deliberately distinct so 'provisional' can never be
# mistaken for 'confirmed' at a glance.
FLIP_STATUS_STYLE: dict[str, dict] = {
    "confirmed": {"label": "Confirmed", "color": "#2E9E5B", "emoji": "✅"},
    "pending":   {"label": "Provisional", "color": "#E8A33D", "emoji": "⏳"},
    "reverted":  {"label": "Reverted", "color": "#9AA0A6", "emoji": "✖"},
}


# ===========================================================================
# Formatting helpers (every causal claim carries its statistic — these keep
# the formatting consistent across the graph, tables and detail panels).
# ===========================================================================

def fmt_p(p: float | None) -> str:
    return f"{p:.2e}" if p is not None else "n/a"


def fmt_corr(c: float | None) -> str:
    return f"{c:+.3f}" if c is not None else "n/a"


def fmt_lag(lag: int | None) -> str:
    return f"{lag}d" if lag is not None else "n/a"


def pair_label(a: str, b: str) -> str:
    return f"{asset_name(a)}  →  {asset_name(b)}"


def asset_class_of(ticker: str) -> AssetClass | None:
    entry = ASSET_UNIVERSE.get(ticker)
    return entry[1] if entry else None


# ===========================================================================
# Cached API reads. Keyed on (base_url, run_id) so a page rerun doesn't refetch
# an 8-year run every interaction. APIError propagates to the caller (cache_data
# does not cache exceptions), which renders a clean error state.
# ===========================================================================

@st.cache_data(show_spinner=False, ttl=30)
def load_run_list(base_url: str) -> list:
    """All persisted runs (most recent first). Short TTL so a freshly-triggered
    analysis appears in the dropdown without a hard reload."""
    return CausalAPIClient(base_url).list_runs()


@st.cache_data(show_spinner=False)
def load_run(base_url: str, run_id: str) -> dict:
    return CausalAPIClient(base_url).get_run(run_id)


@st.cache_data(show_spinner=False)
def load_candidates(base_url: str, run_id: str) -> list:
    return CausalAPIClient(base_url).get_candidates(run_id)


@st.cache_data(show_spinner=False)
def load_graph(base_url: str, run_id: str) -> dict:
    return CausalAPIClient(base_url).get_graph(run_id)


@st.cache_data(show_spinner=False)
def load_regimes(base_url: str, run_id: str) -> list:
    return CausalAPIClient(base_url).get_regimes(run_id)


@st.cache_data(show_spinner=False)
def load_cards(base_url: str, run_id: str) -> list:
    """Layer-2 hypothesis cards (most confident first). Empty list until a
    validation pass has been run for this run."""
    return CausalAPIClient(base_url).get_cards(run_id)


@st.cache_data(show_spinner=False, ttl=30)
def load_llm_health(base_url: str) -> dict:
    """Ollama liveness. Short TTL so the badge reflects the server coming up or
    going down without a full page reload."""
    return CausalAPIClient(base_url).llm_health()


@st.cache_data(show_spinner=False, ttl=30)
def load_flips(base_url: str) -> list:
    """Phase-3 regime-flip events (most recent first), across all monitor runs.
    Short TTL so a freshly-run monitor cycle shows up without a hard reload.
    Not run-scoped: a flip is a diff *between* runs, so the feed is global."""
    return CausalAPIClient(base_url).get_flips()


# ===========================================================================
# Layer-2 helpers — every card is shown WITH its underlying statistic.
# ===========================================================================

def card_pair(card: dict) -> tuple[str, str]:
    """(asset_a, asset_b) for a card — read off the embedded candidate, since
    the API serializes the statistic as a nested ``candidate`` object."""
    cand = card.get("candidate", {})
    return cand.get("asset_a", "?"), cand.get("asset_b", "?")


def flag_meta(flag: str) -> dict:
    return FLAG_STYLE.get(flag, FLAG_STYLE["parse_failed"])


def flag_badge_html(flag: str) -> str:
    m = flag_meta(flag)
    return (
        f"<span style='background:{m['color']};color:#fff;padding:2px 8px;"
        f"border-radius:10px;font-size:12px;font-weight:600'>"
        f"{m['emoji']} {m['label']}</span>"
    )


# ===========================================================================
# Sidebar — connection, run control, load, metadata, filters.
# ===========================================================================

def render_sidebar(client: CausalAPIClient, demo_mode: bool = False) -> dict:
    """Render the whole sidebar; return the active filter selections."""
    st.sidebar.title("⚙️ Controls")

    api_ok = _render_connection(client)
    _render_run_control(client, api_ok, demo_mode)
    _render_load_existing(client, api_ok)
    _render_run_metadata(client)
    _render_layer2_control(client, api_ok, demo_mode)
    return _render_filters(client)


def _render_connection(client: CausalAPIClient) -> bool:
    st.sidebar.caption(f"API: `{client.base_url}`")
    try:
        health = client.health()
    except APIError as exc:
        st.sidebar.error(f"🔴 API offline\n\n{exc}")
        return False
    if health.get("status") == "ok":
        st.sidebar.success("🟢 API online · database reachable")
        return True
    st.sidebar.warning(f"🟠 API degraded: {health}")
    return True


def _render_run_control(
    client: CausalAPIClient, api_ok: bool, demo_mode: bool = False
) -> None:
    if demo_mode:
        # Read-only hosted demo: there is no writable pipeline on the host, so
        # the run-control form is replaced by a clear note rather than shown
        # with a button that would only ever return a 503.
        with st.sidebar.expander("▶️ Run a new analysis", expanded=False):
            st.info(
                "**Demo mode: showing the pre-recorded run only.** Running a "
                "fresh analysis is disabled on the hosted demo — clone the repo "
                "and run it locally (see README) to analyse a live window."
            )
        return
    with st.sidebar.expander("▶️ Run a new analysis", expanded=False):
        st.caption(
            "Runs the full pipeline over all 13 assets. This is **synchronous "
            "and slow** (often several minutes) — for browsing, load an "
            "existing run below instead."
        )
        start = st.date_input(
            "Start date",
            value=_iso_to_date(config.DEFAULT_START_DATE),
            help="Defaults to the config window (spans the 2020 regime break).",
        )
        through_latest = st.checkbox("Through latest available data", value=True)
        end = None
        if not through_latest:
            end = st.date_input("End date", value=_iso_to_date("2026-01-01"))

        max_lag = st.number_input(
            "Max lag (trading days)", min_value=1, max_value=30,
            value=config.DEFAULT_MAX_LAG,
        )
        alpha = st.slider(
            "Alpha (significance)", min_value=0.01, max_value=0.10,
            value=config.DEFAULT_ALPHA, step=0.01,
        )
        method = st.selectbox(
            "Multiple-comparisons correction",
            options=["fdr_bh", "bonferroni"],
            index=0 if config.DEFAULT_CORRECTION == "fdr_bh" else 1,
        )

        if st.button("Run analysis", type="primary", disabled=not api_ok):
            body = {
                "start_date": start.isoformat(),
                "end_date": end.isoformat() if end else None,
                "max_lag": int(max_lag),
                "alpha": float(alpha),
                "correction_method": method,
            }
            with st.spinner("Running the full pipeline — this can take minutes…"):
                try:
                    run = client.analyze(body)
                except APIError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["run_id"] = run["run_id"]
                    st.success(f"Run complete: {run['run_id']}")
                    st.rerun()


def _render_load_existing(client: CausalAPIClient, api_ok: bool) -> None:
    with st.sidebar.expander("📂 Load an existing run", expanded=True):
        runs: list[dict] = []
        if api_ok:
            try:
                runs = load_run_list(client.base_url)
            except APIError:
                runs = []

        if runs:
            # Build label: "YYYY-MM-DD → YYYY-MM-DD  (run_id)"
            def _run_label(r: dict) -> str:
                return (
                    f"{r['start_date']} → {r['end_date']}  "
                    f"({r['run_id']})"
                )

            options = {_run_label(r): r["run_id"] for r in runs}
            current_id = st.session_state.get("run_id", "")
            # Find current label so the selectbox lands on it after a rerun.
            current_label = next(
                (lbl for lbl, rid in options.items() if rid == current_id), None
            )
            idx = list(options).index(current_label) if current_label else 0

            choice = st.selectbox(
                "Select run",
                options=list(options.keys()),
                index=idx,
                help="Most recent run first. Select, then click Load.",
            )
            if st.button("Load run", disabled=not api_ok):
                rid = options[choice]
                st.session_state["run_id"] = rid
                st.rerun()
        else:
            # Fallback: no runs discovered — let the user paste an id directly.
            st.caption(
                "No runs found in the API yet. "
                "Run an analysis from the panel above, or paste a run_id:"
            )
            run_id = st.text_input(
                "run_id",
                value=st.session_state.get("run_id", ""),
                placeholder="run_YYYYMMDD_…",
            )
            if st.button("Load run", disabled=not api_ok):
                rid = run_id.strip()
                if not rid:
                    st.warning("Enter a run_id first.")
                else:
                    try:
                        client.get_run(rid)
                    except APIError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["run_id"] = rid
                        st.rerun()


def _render_run_metadata(client: CausalAPIClient) -> None:
    run_id = st.session_state.get("run_id")
    if not run_id:
        return
    try:
        run = load_run(client.base_url, run_id)
        candidates = load_candidates(client.base_url, run_id)
    except APIError as exc:
        st.sidebar.error(str(exc))
        return

    n_total = len(candidates)
    n_sig = sum(1 for c in candidates if c.get("is_significant"))

    st.sidebar.divider()
    st.sidebar.subheader("Loaded run")
    st.sidebar.caption(f"`{run_id}`")
    st.sidebar.write(f"**Window:** {run['start_date']} → {run['end_date']}")
    st.sidebar.write(
        f"**Correction:** {run['correction_method']} · α = {run['alpha']}"
    )
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("Assets", len(run["asset_universe"]))
    col2.metric("Candidates", n_total)
    col3.metric("Significant", n_sig)


def _render_layer2_control(
    client: CausalAPIClient, api_ok: bool, demo_mode: bool = False
) -> None:
    """Ollama health + a (slow) on-demand validation trigger for the loaded run.

    Degrades gracefully: if Ollama is unreachable (or in read-only demo mode)
    the *generate* action is disabled with a clear message, never a stack trace.
    Reading the pre-generated cards always works."""
    with st.sidebar.expander("🧠 Layer 2 (LLM plausibility)", expanded=False):
        run_id = st.session_state.get("run_id")
        # Ollama liveness badge.
        health = None
        if api_ok:
            try:
                health = load_llm_health(client.base_url)
            except APIError as exc:
                st.caption(f"Could not check the model: {exc}")
        ollama_up = bool(health and health.get("ollama_available"))
        if demo_mode:
            st.info(
                "**Demo mode:** card *generation* is disabled (the hosted demo "
                "has no local LLM). The pre-recorded run's hypothesis cards are "
                "already loaded — browse them in the **🧠 Hypothesis cards** tab."
            )
        elif health is None:
            st.warning("LLM status unknown (API offline).")
        elif ollama_up:
            st.success(f"🟢 Ollama online · `{health.get('model_name','?')}`")
        else:
            st.warning(
                "🟠 Ollama not reachable. Start it with `ollama serve` and "
                f"pull `{health.get('model_name','the model')}`. You can still "
                "browse previously-generated cards."
            )

        st.caption(
            "Generates a hypothesis card per **significant** candidate. The LLM "
            "only *explains* a finding that already passed significance + "
            "correction — it never re-tests it or upgrades it to causal."
        )
        n_cap = st.number_input(
            "Validate at most N (by significance)",
            min_value=1, max_value=200, value=10,
            help="A local 8B model takes ~1 min per candidate, so cap it for "
                 "interactive use. The recorded results/ run validated all 106.",
        )
        disabled = demo_mode or not (api_ok and ollama_up and run_id)
        if st.button("Generate hypothesis cards", disabled=disabled):
            with st.spinner(
                f"Validating up to {int(n_cap)} candidates — ~1 min each…"
            ):
                try:
                    summary = client.validate(run_id, limit=int(n_cap))
                except APIError as exc:
                    st.error(str(exc))
                else:
                    load_cards.clear()  # invalidate the cached card list
                    st.success(
                        f"Generated {summary['n_candidates_validated']} cards: "
                        + ", ".join(
                            f"{k}={v}" for k, v in summary["counts"].items() if v
                        )
                    )
                    st.rerun()
        if not run_id:
            st.caption("Load a run first to enable validation.")


def _render_filters(client: CausalAPIClient) -> dict:
    """Global filters applied to the graph, exploration and card views."""
    st.sidebar.divider()
    st.sidebar.subheader("🔎 Filters")

    run = None
    run_id = st.session_state.get("run_id")
    if run_id:
        try:
            run = load_run(client.base_url, run_id)
        except APIError:
            run = None
    alpha = float(run["alpha"]) if run else config.DASHBOARD_DEFAULT_SIG_THRESHOLD

    sig_threshold = st.sidebar.slider(
        "Corrected p-value ≤",
        min_value=0.0, max_value=1.0,
        value=min(alpha, 1.0), step=0.005, format="%.3f",
        help="Keep only relationships at least this significant after "
             "multiple-comparisons correction.",
    )
    class_options = list(AssetClass)
    selected_classes = st.sidebar.multiselect(
        "Asset class",
        options=class_options,
        default=class_options,
        format_func=lambda c: c.value.replace("_", " ").title(),
        help="Restrict the universe shown in the graph and exploration views.",
    )

    # Plausibility-flag filter (applies to the Layer-2 card feed). Shown always
    # so it's discoverable; it only bites once cards exist.
    selected_flags = st.sidebar.multiselect(
        "LLM plausibility flag",
        options=FLAG_ORDER,
        default=FLAG_ORDER,
        format_func=lambda f: FLAG_STYLE[f]["label"],
        help="Filter the hypothesis-card feed by the LLM's heuristic "
             "plausibility judgement. Remember: this is a filter, not proof.",
    )
    return {
        "sig_threshold": sig_threshold,
        "classes": set(selected_classes) if selected_classes else set(class_options),
        "flags": set(selected_flags) if selected_flags else set(FLAG_ORDER),
    }


def _iso_to_date(iso: str):
    from datetime import date

    return date.fromisoformat(iso)


# ===========================================================================
# View 1 — Causal graph (static, interactive pan/zoom/hover).
# ===========================================================================

def edge_inline_label(e: dict) -> str:
    """Compact label drawn on the edge — the honesty rule made visible: a
    corrected p-value, lag and correlation accompany every arrow."""
    return (
        f"p={fmt_p(e['corrected_p_value'])} · "
        f"lag {fmt_lag(e.get('lag'))} · r={fmt_corr(e.get('correlation_strength'))}"
    )


def edge_hover(e: dict) -> str:
    return (
        f"{pair_label(e['source'], e['target'])}\n"
        f"type: {e['edge_type']} (oriented by {e['orientation_source']})\n"
        f"corrected p: {fmt_p(e['corrected_p_value'])}\n"
        f"lag: {fmt_lag(e.get('lag'))}\n"
        f"correlation: {fmt_corr(e.get('correlation_strength'))}"
    )


def build_agraph(graph: dict, filters: dict) -> tuple[list[Node], list[Edge], int]:
    classes = filters["classes"]
    threshold = filters["sig_threshold"]

    visible_nodes = [
        t for t in graph["nodes"]
        if asset_class_of(t) is None or asset_class_of(t) in classes
    ]
    visible_set = set(visible_nodes)

    edges: list[Edge] = []
    for e in graph["edges"]:
        if e["corrected_p_value"] > threshold:
            continue
        if e["source"] not in visible_set or e["target"] not in visible_set:
            continue
        style = EDGE_STYLE.get(e["edge_type"], EDGE_STYLE["undirected"])
        kwargs = dict(
            source=e["source"],
            target=e["target"],
            label=edge_inline_label(e),
            title=edge_hover(e),
            color=style["color"],
            dashes=style["dashes"],
            width=2,
            font={"size": 9, "align": "top", "color": "#c9ccd1"},
            smooth={"type": "curvedCW", "roundness": 0.15},
        )
        if style["arrows"]:
            kwargs["arrows"] = style["arrows"]
        edges.append(Edge(**kwargs))

    nodes = [
        Node(
            id=t,
            label=asset_name(t),
            title=f"{asset_name(t)} ({t})",
            color=ASSET_CLASS_COLOR.get(asset_class_of(t), "#888888"),
            size=22,
            font={"color": "#f0f0f0", "size": 14},
        )
        for t in visible_nodes
    ]
    return nodes, edges, len(edges)


def render_graph_legend() -> None:
    edge_rows = [
        ("Directed (PC-oriented)", EDGE_STYLE["directed"]["color"],
         "A → B: arrow direction is the discovered orientation."),
        ("Undirected (Granger-significant)", EDGE_STYLE["undirected"]["color"],
         "Dashed, no arrowhead: link is significant but PC left it un-oriented."),
        ("Bidirected", EDGE_STYLE["bidirected"]["color"],
         "Arrows both ends: mutual / possible latent confounder."),
    ]
    edge_html = "".join(
        f"<div style='margin:2px 0'>"
        f"<span style='display:inline-block;width:26px;height:0;"
        f"border-top:3px {'dashed' if 'Undirected' in name else 'solid'} {color};"
        f"vertical-align:middle'></span>&nbsp; "
        f"<b>{name}</b> — <span style='color:#9aa0a6'>{desc}</span></div>"
        for name, color, desc in edge_rows
    )
    node_html = "".join(
        f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
        f"background:{ASSET_CLASS_COLOR[c]};margin:0 4px 0 10px'></span>"
        f"<span style='color:#c9ccd1'>{c.value.replace('_',' ').title()}</span>"
        for c in AssetClass
    )
    st.markdown("**Edge encoding**", help="Every edge also carries its statistic.")
    st.markdown(edge_html, unsafe_allow_html=True)
    st.markdown("**Node colour = asset class**", unsafe_allow_html=True)
    st.markdown(node_html, unsafe_allow_html=True)


def view_causal_graph(client: CausalAPIClient, run_id: str, filters: dict) -> None:
    st.subheader("Discovered causal graph")
    st.caption(
        "Nodes are the 13 assets; edges are the relationships the PC algorithm "
        "kept from the significant Granger candidates. Drag to pan, scroll to "
        "zoom, hover an edge for full statistics."
    )
    try:
        graph = load_graph(client.base_url, run_id)
    except APIError as exc:
        st.error(str(exc))
        return

    nodes, edges, n_shown = build_agraph(graph, filters)
    total_edges = len(graph["edges"])
    st.caption(
        f"Showing **{n_shown}** of {total_edges} graph edges "
        f"(corrected p ≤ {filters['sig_threshold']:.3f}, "
        f"{len(nodes)} of {len(graph['nodes'])} assets in selected classes)."
    )
    if n_shown < total_edges and len(nodes) == len(graph["nodes"]):
        st.caption(
            "ℹ️ PC can retain a structural edge whose *Granger* corrected "
            "p-value still exceeds the threshold. Raise the slider toward 1.0 "
            "to reveal the full discovered skeleton — each edge shows its own "
            "p-value, so weak links are never disguised as strong ones."
        )

    if not edges:
        st.info(
            "No edges match the current filters. Loosen the corrected p-value "
            "threshold or widen the asset-class selection in the sidebar."
        )

    cfg = Config(
        height=config.DASHBOARD_GRAPH_HEIGHT,
        width=config.DASHBOARD_GRAPH_WIDTH,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#FFD166",
        collapsible=False,
    )
    graph_col, legend_col = st.columns([3, 1])
    with graph_col:
        clicked = agraph(nodes=nodes, edges=edges, config=cfg)
    with legend_col:
        render_graph_legend()

    if clicked:
        st.caption(
            f"Selected **{asset_name(clicked)}** ({clicked}). "
            "Open the *Explore* tab to inspect its in/out edges."
        )


# ===========================================================================
# View 2 — Regime timeline (static, per pair).
# ===========================================================================

def regime_timeline(periods: list[dict], title: str) -> None:
    """Render a pair's time-bound regime windows as a two-row timeline."""
    if not periods:
        st.info("No regime windows were computed for this pair.")
        return

    df = pl.DataFrame(periods).with_columns(
        pl.col("start").str.to_date(),
        pl.col("end").str.to_date(),
        pl.when(pl.col("active"))
        .then(pl.lit("Active (coupled)"))
        .otherwise(pl.lit("Inactive (decoupled)"))
        .alias("state"),
    )

    state_scale = alt.Scale(
        domain=["Active (coupled)", "Inactive (decoupled)"],
        range=[ACTIVE_COLOR, INACTIVE_COLOR],
    )
    tooltip = [
        alt.Tooltip("start:T", title="From"),
        alt.Tooltip("end:T", title="To"),
        alt.Tooltip("state:N", title="State"),
        alt.Tooltip("mean_correlation:Q", title="Mean corr", format="+.3f"),
    ]

    lanes = (
        alt.Chart(df)
        .mark_bar(height=22, cornerRadius=2)
        .encode(
            x=alt.X("start:T", title=None),
            x2="end:T",
            y=alt.Y("state:N", title=None, sort=list(state_scale.domain)),
            color=alt.Color("state:N", scale=state_scale,
                            legend=alt.Legend(title="Regime", orient="top")),
            tooltip=tooltip,
        )
        .properties(height=90, title=title)
    )

    corr = (
        alt.Chart(df)
        .mark_bar(cornerRadius=1)
        .encode(
            x=alt.X("start:T", title="Date"),
            x2="end:T",
            y=alt.Y("mean_correlation:Q", title="Mean corr"),
            color=alt.condition(
                alt.datum.mean_correlation >= 0,
                alt.value(POS_CORR_COLOR), alt.value(NEG_CORR_COLOR),
            ),
            tooltip=tooltip,
        )
        .properties(height=140)
    )

    st.altair_chart(
        alt.vconcat(lanes, corr).resolve_scale(x="shared"),
        width="stretch",
    )


def view_regime_timeline(client: CausalAPIClient, run_id: str) -> None:
    st.subheader("Regime timeline")
    st.warning(
        "⏳ **Regimes are time-bound, not permanent.** Each window says the "
        "relationship was (in)active over *that date range only* — a coupling "
        "that held in 2020 is not a standing claim about today.",
        icon="⏳",
    )
    try:
        regimes = load_regimes(client.base_url, run_id)
    except APIError as exc:
        st.error(str(exc))
        return
    if not regimes:
        st.info("This run has no regime windows (no significant pairs).")
        return

    options = {pair_label(r["asset_a"], r["asset_b"]): r for r in regimes}
    choice = st.selectbox("Asset pair", options=list(options.keys()))
    selected = options[choice]
    periods = selected["regime_periods"]

    active = sum(1 for p in periods if p["active"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Windows", len(periods))
    c2.metric("Active", active)
    c3.metric("Inactive", len(periods) - active)

    regime_timeline(periods, title=choice)


# ===========================================================================
# View 3 — Interactive exploration (node-centric + edge-centric).
# ===========================================================================

def _candidates_to_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "Driver": asset_name(c["asset_a"]),
                "Affected": asset_name(c["asset_b"]),
                "Lag (d)": c["lag"],
                "Corrected p": c["corrected_p_value"],
                "Raw p": c["granger_p_value"],
                "Correlation": c["correlation_strength"],
                "Confidence": c["statistical_confidence"],
                "Significant": c["is_significant"],
            }
            for c in rows
        ]
    )


def _filter_candidates(candidates: list[dict], filters: dict) -> list[dict]:
    threshold = filters["sig_threshold"]
    classes = filters["classes"]
    out = []
    for c in candidates:
        if c["corrected_p_value"] > threshold:
            continue
        ca, cb = asset_class_of(c["asset_a"]), asset_class_of(c["asset_b"])
        if (ca is not None and ca not in classes) or (cb is not None and cb not in classes):
            continue
        out.append(c)
    return out


def view_explore(client: CausalAPIClient, run_id: str, filters: dict) -> None:
    st.subheader("Interactive exploration")
    try:
        candidates = load_candidates(client.base_url, run_id)
    except APIError as exc:
        st.error(str(exc))
        return

    shown = _filter_candidates(candidates, filters)
    st.caption(
        f"{len(shown)} of {len(candidates)} candidates pass the current filters "
        f"(corrected p ≤ {filters['sig_threshold']:.3f})."
    )

    # Layer-2 cards, keyed by candidate_id, so the edge-detail panel can show the
    # real hypothesis card instead of the old placeholder. Empty if none yet.
    try:
        cards = load_cards(client.base_url, run_id)
    except APIError:
        cards = []
    cards_by_candidate = {
        c.get("candidate", {}).get("candidate_id"): c for c in cards
    }

    node_tab, edge_tab = st.tabs(["By asset", "By relationship"])
    with node_tab:
        _explore_by_asset(shown, filters)
    with edge_tab:
        _explore_by_edge(shown, cards_by_candidate)


def _explore_by_asset(candidates: list[dict], filters: dict) -> None:
    assets = sorted(
        {t for t in ASSET_UNIVERSE if asset_class_of(t) in filters["classes"]},
        key=asset_name,
    )
    if not assets:
        st.info("No assets in the selected classes.")
        return
    asset = st.selectbox("Asset", options=assets, format_func=asset_name)

    outgoing = [c for c in candidates if c["asset_a"] == asset]
    incoming = [c for c in candidates if c["asset_b"] == asset]

    st.markdown(f"**{asset_name(asset)}** drives → *(outgoing)*")
    if outgoing:
        st.dataframe(_candidates_to_df(outgoing), width="stretch",
                     hide_index=True)
    else:
        st.caption("No outgoing relationships pass the filters.")

    st.markdown(f"**{asset_name(asset)}** is driven by ← *(incoming)*")
    if incoming:
        st.dataframe(_candidates_to_df(incoming), width="stretch",
                     hide_index=True)
    else:
        st.caption("No incoming relationships pass the filters.")


def _explore_by_edge(
    candidates: list[dict], cards_by_candidate: dict[str, dict] | None = None
) -> None:
    cards_by_candidate = cards_by_candidate or {}
    if not candidates:
        st.info("No relationships pass the current filters.")
        return
    options = {
        f"{pair_label(c['asset_a'], c['asset_b'])}   "
        f"(p={fmt_p(c['corrected_p_value'])})": c
        for c in candidates
    }
    choice = st.selectbox("Relationship", options=list(options.keys()))
    c = options[choice]

    st.markdown(f"### {pair_label(c['asset_a'], c['asset_b'])}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Corrected p", fmt_p(c["corrected_p_value"]))
    m2.metric("Raw Granger p", fmt_p(c["granger_p_value"]))
    m3.metric("Lag", fmt_lag(c["lag"]))
    m4.metric("Correlation", fmt_corr(c["correlation_strength"]))

    e1, e2 = st.columns(2)
    e1.metric("Statistical confidence", f"{c['statistical_confidence']:.2f}")
    e2.metric("Significant?", "yes" if c["is_significant"] else "no")
    st.caption(
        f"Direction: {asset_name(c['asset_a'])} Granger-precedes "
        f"{asset_name(c['asset_b'])} at lag {fmt_lag(c['lag'])}. "
        "Granger causality is predictive precedence, **not** proof of causation."
    )

    st.markdown("#### Regime history")
    regime_timeline(
        c.get("regime_periods", []),
        title=pair_label(c["asset_a"], c["asset_b"]),
    )

    st.markdown("#### Economic mechanism & plausibility")
    card = cards_by_candidate.get(c.get("candidate_id"))
    if card is not None:
        st.caption(
            "LLM hypothesis card for this relationship. The plausibility flag is "
            "a **heuristic filter, not validation** — the model can rationalise a "
            "spurious finding. The corrected p-value above is the evidence."
        )
        render_card(card, show_pair_header=False)
    else:
        st.info(
            "No Layer-2 hypothesis card for this relationship yet. Generate cards "
            "from the **🧠 Layer 2 (LLM plausibility)** panel in the sidebar "
            "(needs a running Ollama). The card adds a *proposed* economic "
            "mechanism and plausibility flag on top of the statistic shown above.",
            icon="🧠",
        )


# ===========================================================================
# View 4 — Layer-2 hypothesis-card feed.
# ===========================================================================

def render_card(card: dict, *, show_pair_header: bool = True) -> None:
    """Render one hypothesis card WITH its underlying statistic (hard rule: no
    card is ever shown without the corrected p-value that justifies it)."""
    cand = card.get("candidate", {})
    a, b = card_pair(card)
    flag = card.get("plausibility_flag", "parse_failed")

    if show_pair_header:
        st.markdown(f"##### {pair_label(a, b)}")
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown(flag_badge_html(flag), unsafe_allow_html=True)
        if card.get("mechanism_channel"):
            st.caption(f"Channel: **{card['mechanism_channel']}**")
    with cols[1]:
        st.metric("LLM confidence", f"{card.get('llm_confidence', 0):.2f}")

    st.write(card.get("mechanism_explanation", ""))

    caveats = card.get("caveats") or []
    if caveats:
        st.markdown("**Caveats / alternative explanations:**")
        for cv in caveats:
            st.markdown(f"- {cv}")

    if not card.get("in_graph", True):
        flagline = (
            "engaged with the rejection ✓" if card.get("addresses_pc_rejection")
            else "did **not** explicitly address the rejection"
        )
        st.caption(
            f"⚠️ PC rejected this as a *direct* edge (in_graph = false). The card "
            f"{flagline}. A Granger-strong signal that PC drops is usually "
            "mediated or confounded — not a direct link."
        )

    # The statistic — always attached.
    st.caption(
        f"**Underlying statistic** · corrected p = {fmt_p(cand.get('corrected_p_value'))} · "
        f"raw Granger p = {fmt_p(cand.get('granger_p_value'))} · "
        f"lag {fmt_lag(cand.get('lag'))} · r = {fmt_corr(cand.get('correlation_strength'))} · "
        f"statistical confidence = {cand.get('statistical_confidence', float('nan')):.2f} · "
        f"in PC graph: {'yes' if card.get('in_graph') else 'no'}"
    )
    st.divider()


def _filter_cards(cards: list[dict], filters: dict) -> list[dict]:
    out = []
    for c in cards:
        if c.get("plausibility_flag") not in filters["flags"]:
            continue
        cand = c.get("candidate", {})
        if cand.get("corrected_p_value", 1.0) > filters["sig_threshold"]:
            continue
        a, b = card_pair(c)
        ca, cb = asset_class_of(a), asset_class_of(b)
        if (ca is not None and ca not in filters["classes"]) or (
            cb is not None and cb not in filters["classes"]
        ):
            continue
        out.append(c)
    return out


def view_hypothesis_cards(client: CausalAPIClient, run_id: str, filters: dict) -> None:
    st.subheader("LLM hypothesis cards")
    st.warning(
        "🧪 **The plausibility flag is a heuristic filter, not validation.** A "
        "language model can produce a confident, fluent economic mechanism for a "
        "statistically **spurious** relationship. Treat 'plausible' as 'worth a "
        "human look', never as proof — and note every card still carries its "
        "corrected p-value.",
        icon="🧪",
    )
    try:
        cards = load_cards(client.base_url, run_id)
    except APIError as exc:
        st.error(str(exc))
        return

    if not cards:
        st.info(
            "No hypothesis cards for this run yet. Generate them from the "
            "**🧠 Layer 2 (LLM plausibility)** panel in the sidebar (needs a "
            "running Ollama). This is slow — a local 8B model takes ~1 min per "
            "candidate."
        )
        return

    # Flag counts across ALL cards for the run (not just the filtered view).
    counts = {f: 0 for f in FLAG_ORDER}
    for c in cards:
        counts[c.get("plausibility_flag", "parse_failed")] = (
            counts.get(c.get("plausibility_flag", "parse_failed"), 0) + 1
        )
    cols = st.columns(len(FLAG_ORDER))
    for col, f in zip(cols, FLAG_ORDER):
        col.metric(f"{FLAG_STYLE[f]['emoji']} {FLAG_STYLE[f]['label']}", counts[f])

    shown = _filter_cards(cards, filters)
    st.caption(
        f"Showing **{len(shown)}** of {len(cards)} cards "
        f"(sorted by LLM confidence; filters from the sidebar applied)."
    )
    if not shown:
        st.info("No cards match the current filters.")
        return
    for c in shown:
        render_card(c)


# ===========================================================================
# View 5 — Business use cases (the cards + stats mapped to decisions).
# ===========================================================================

def _regime_flips(cand: dict) -> int:
    """Number of active/inactive transitions in a candidate's regime history —
    a proxy for how 'switchy' (regime-dependent) the relationship is."""
    periods = cand.get("regime_periods") or []
    states = [bool(p.get("active")) for p in periods]
    return sum(1 for i in range(1, len(states)) if states[i] != states[i - 1])


def view_business_use_cases(client: CausalAPIClient, run_id: str) -> None:
    st.subheader("Business use cases")
    st.caption(
        "How the current findings map to four screening use cases. Every item "
        "below is a **candidate for human review**, carrying its statistic — not "
        "a recommendation, and certainly not a trade."
    )
    try:
        cards = load_cards(client.base_url, run_id)
    except APIError as exc:
        st.error(str(exc))
        return
    if not cards:
        st.info(
            "Generate the Layer-2 hypothesis cards first (sidebar) — the use-case "
            "views below are built from the cards plus their statistics."
        )
        return

    plausible = [
        c for c in cards
        if c.get("plausibility_flag")
        in ("plausible_known_mechanism", "plausible_novel")
    ]

    # 1) Hedging timing — known-mechanism leads with a non-zero lag give a window.
    st.markdown("#### 1 · Hedging timing")
    st.caption(
        "Directional leads with a *named* mechanism and a lag suggest a window: "
        "the driver tends to move first, so its move is an early read on the "
        "affected asset. The lag is the timing; the corrected p-value is the "
        "evidence. (Predictive precedence, not a guarantee.)"
    )
    timing = sorted(
        [c for c in cards if c.get("plausibility_flag") == "plausible_known_mechanism"],
        key=lambda c: c.get("llm_confidence", 0), reverse=True,
    )
    if timing:
        rows = [
            {
                "Driver leads": asset_name(card_pair(c)[0]),
                "Affected": asset_name(card_pair(c)[1]),
                "Lead (days)": c["candidate"].get("lag"),
                "Channel": c.get("mechanism_channel") or "—",
                "Corrected p": c["candidate"].get("corrected_p_value"),
                "LLM conf.": c.get("llm_confidence"),
            }
            for c in timing[:10]
        ]
        st.dataframe(pl.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.caption("No PLAUSIBLE_KNOWN_MECHANISM cards in this run.")

    # 2) Hidden concentration risk — one driver fanning out to many names.
    st.markdown("#### 2 · Hidden concentration risk")
    st.caption(
        "If a single driver plausibly leads many otherwise-separate assets, "
        "positions that look diversified may share one hidden factor. Counts use "
        "cards the LLM did **not** flag spurious."
    )
    fan: dict[str, int] = {}
    for c in plausible:
        a = card_pair(c)[0]
        fan[a] = fan.get(a, 0) + 1
    hubs = sorted(fan.items(), key=lambda kv: kv[1], reverse=True)
    hubs = [(a, n) for a, n in hubs if n >= 2]
    if hubs:
        st.dataframe(
            pl.DataFrame(
                [{"Driver": asset_name(a), "# plausible downstream links": n}
                 for a, n in hubs[:10]]
            ),
            width="stretch", hide_index=True,
        )
    else:
        st.caption("No driver has ≥2 non-spurious downstream links in this run.")

    # 3) Macro narrative validation — LLM narrative vs the number.
    st.markdown("#### 3 · Macro narrative validation")
    st.caption(
        "Where the model proposes a macro channel, you can sanity-check the "
        "*story* against the *statistic*. Especially useful where the LLM "
        "**disagrees** with a strong number (flags it spurious) — that tension "
        "is the screen, not the answer."
    )
    tension = [
        c for c in cards
        if c.get("plausibility_flag") == "likely_spurious"
        and c["candidate"].get("corrected_p_value", 1.0) < 1e-6
    ]
    if tension:
        st.markdown(
            "**Statistically strong, yet the LLM withholds a mechanism** "
            "(corrected p < 1e-6 but flagged likely-spurious):"
        )
        st.dataframe(
            pl.DataFrame(
                [
                    {
                        "Relationship": pair_label(*card_pair(c)),
                        "Corrected p": c["candidate"].get("corrected_p_value"),
                        "In PC graph": c.get("in_graph"),
                        "LLM note": (c.get("caveats") or ["—"])[0],
                    }
                    for c in sorted(
                        tension,
                        key=lambda c: c["candidate"].get("corrected_p_value", 1.0),
                    )[:8]
                ]
            ),
            width="stretch", hide_index=True,
        )
    else:
        st.caption("No strong-but-spurious tensions surfaced in this run.")

    # 4) Regime-change early warning — the switchiest couplings.
    st.markdown("#### 4 · Regime-change early warning")
    st.caption(
        "Relationships whose coupling **switches on and off** most often are the "
        "ones to watch for regime change — a coupling that was inactive turning "
        "active (or vice-versa) is the early-warning signal. Regimes are "
        "time-bound, never permanent."
    )
    switchy = sorted(
        cards, key=lambda c: _regime_flips(c["candidate"]), reverse=True
    )
    switchy = [c for c in switchy if _regime_flips(c["candidate"]) > 0]
    if switchy:
        st.dataframe(
            pl.DataFrame(
                [
                    {
                        "Relationship": pair_label(*card_pair(c)),
                        "Regime switches": _regime_flips(c["candidate"]),
                        "Plausibility": FLAG_STYLE.get(
                            c.get("plausibility_flag"), FLAG_STYLE["parse_failed"]
                        )["label"],
                        "Corrected p": c["candidate"].get("corrected_p_value"),
                    }
                    for c in switchy[:10]
                ]
            ),
            width="stretch", hide_index=True,
        )
    else:
        st.caption("No regime history available for this run's cards.")


# ===========================================================================
# View 6 — Events: scheduled regime-flip monitoring feed (Phase 3).
# ===========================================================================

def flip_status_meta(status: str) -> dict:
    return FLIP_STATUS_STYLE.get(status, FLIP_STATUS_STYLE["reverted"])


def flip_badge_html(ev: dict) -> str:
    """A prominent, colour-distinct pill for a flip's lifecycle state. A pending
    flip shows its confirmation progress (n/N) inside the badge so 'provisional'
    is impossible to miss — never just a quiet text label."""
    status = ev.get("status", "reverted")
    m = flip_status_meta(status)
    extra = ""
    if status == "pending":
        n = ev.get("consecutive_confirmations", 1)
        extra = f" · {n}/{config.MONITOR_CONFIRMATION_RUNS}"
    return (
        f"<span style='background:{m['color']};color:#fff;padding:3px 12px;"
        f"border-radius:11px;font-size:12px;font-weight:700;letter-spacing:.4px'>"
        f"{m['emoji']} {m['label'].upper()}{extra}</span>"
    )


def flip_stat_line(ev: dict) -> str:
    """The hard rule, carried into Phase 3: no flip is ever shown without the
    statistic of its new-run candidate."""
    return (
        f"**Underlying statistic** · corrected p = {fmt_p(ev.get('corrected_p_value'))} · "
        f"lag {fmt_lag(ev.get('lag'))} · "
        f"r = {fmt_corr(ev.get('correlation_strength'))}"
    )


def render_flip_event(ev: dict) -> None:
    """One regime-flip event: pair, lifecycle badge, transition, and — always —
    its underlying statistic."""
    a, b = ev.get("asset_a", "?"), ev.get("asset_b", "?")
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown(f"##### {pair_label(a, b)}")
        st.markdown(flip_badge_html(ev), unsafe_allow_html=True)
        st.caption(f"Transition: {ev.get('direction', '?')}")
    with cols[1]:
        if ev.get("status") == "pending":
            st.metric(
                "Confirmations",
                f"{ev.get('consecutive_confirmations', 1)}"
                f"/{config.MONITOR_CONFIRMATION_RUNS}",
                help="Consecutive monitor runs the new status has held. The flip "
                     "is 'confirmed' only once this reaches the full window.",
            )
        elif ev.get("status") == "confirmed":
            st.metric("Confirmed over", f"{config.MONITOR_CONFIRMATION_RUNS} runs")

    st.caption(flip_stat_line(ev))
    detected = (ev.get("detected_at") or "")[:19].replace("T", " ")
    st.caption(
        f"Detected {detected} UTC · prior run `{ev.get('prior_run_id', '?')}` → "
        f"new run `{ev.get('new_run_id', '?')}`"
    )
    st.divider()


def view_events(client: CausalAPIClient, demo_mode: bool = False) -> None:
    st.subheader("Regime-flip events")
    if demo_mode:
        st.caption(
            "🔭 Demo mode: this feed is read-only. The pre-recorded analysis run "
            "has no flips of its own (flips are diffs *between* monitor runs); "
            "triggering a live monitor cycle requires running the engine locally."
        )
    st.warning(
        "🚨 **Flips are detected against *historical* data and can be reverted by "
        "data revisions.** yfinance can restate the most recent bars after the "
        "fact, so a fresh flip may be an artifact. **'Confirmed' here means the "
        "new regime status stayed stable across "
        f"{config.MONITOR_CONFIRMATION_RUNS} consecutive monitor runs — it does "
        "NOT mean the relationship was validated causally.** Provisional "
        "(pending) flips have not cleared that bar yet and may still revert; "
        "treat them as early, unconfirmed signals only.",
        icon="🚨",
    )
    try:
        flips = load_flips(client.base_url)
    except APIError as exc:
        st.error(str(exc))
        return

    if not flips:
        if demo_mode:
            st.info(
                "No regime-flip events in the pre-recorded run shown by this "
                "demo. Flips are diffs *between* successive monitor runs, which "
                "the hosted read-only demo does not run. Clone the repo and run "
                "`python -m scripts.run_monitor` locally (twice or more) to "
                "populate this feed — see the README for the recorded "
                "4-snapshot validation (10 confirmed + 28 pending flips)."
            )
            return
        st.info(
            "No regime-flip events recorded yet. The monitor populates this feed "
            "by diffing successive runs: run `python -m scripts.run_monitor` (or "
            "`POST /monitor`) at least **twice** against later data, or schedule "
            "it daily (Windows Task Scheduler). The first run only establishes a "
            "baseline — flips appear from the second cycle onward."
        )
        return

    confirmed = [f for f in flips if f.get("status") == "confirmed"]
    pending = [f for f in flips if f.get("status") == "pending"]
    reverted = [f for f in flips if f.get("status") == "reverted"]

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Confirmed", len(confirmed))
    c2.metric("⏳ Provisional", len(pending))
    c3.metric("✖ Reverted", len(reverted))

    # Confirmed first and most prominent — the real signal.
    st.markdown("### ✅ Confirmed regime changes")
    st.caption(
        f"Survived {config.MONITOR_CONFIRMATION_RUNS} consecutive monitor runs "
        "without snapping back. These are the signals worth a human look — still "
        "predictive precedence, not proof of causation."
    )
    if confirmed:
        for f in confirmed:
            render_flip_event(f)
    else:
        st.info("No flips have been confirmed yet.")

    # Pending — clearly demarcated as provisional.
    st.markdown("### ⏳ Provisional — pending confirmation")
    st.caption(
        "Detected, but **not yet** stable across the confirmation window. May "
        "still revert (e.g. a data revision). Not a finding yet — shown so you "
        "can watch them, with their progress toward confirmation on each badge."
    )
    if pending:
        for f in pending:
            render_flip_event(f)
    else:
        st.caption("Nothing pending.")

    # Reverted — tucked away; recorded for honesty, not surfaced as signal.
    if reverted:
        with st.expander(
            f"✖ Reverted ({len(reverted)}) — snapped back / rejected as transient",
            expanded=False,
        ):
            st.caption(
                "These flips reverted before confirming (or their pair lost "
                "significance). Kept for transparency — never presented as a "
                "finding."
            )
            for f in reverted:
                render_flip_event(f)


# ===========================================================================
# App shell.
# ===========================================================================

def is_demo_mode(client: CausalAPIClient) -> bool:
    """Detect a read-only hosted deployment.

    The API's ``/health`` is the single source of truth (so detection works no
    matter where the dashboard itself is deployed — e.g. Streamlit Cloud talking
    to a Render API). Falls back to a local ``DEMO_MODE`` env var if the API
    can't be reached, so the banner still shows during a brief API outage."""
    try:
        health = client.health()
        if isinstance(health, dict) and "demo_mode" in health:
            return bool(health["demo_mode"])
    except APIError:
        pass
    return os.environ.get("DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def render_demo_banner() -> None:
    """Prominent top-of-page banner for the hosted read-only demo."""
    st.info(
        "Live read-only demo — showing a pre-recorded 8-year analysis. The "
        "local-LLM layer and live monitoring require running this locally "
        "(see README).",
        icon="🔭",
    )


def render_disclaimer() -> None:
    st.error(
        "**Read this first.** Granger causality measures *predictive "
        "precedence*, not proof of causation. Everything shown here is a "
        "**candidate hypothesis for human review**, surfaced with its corrected "
        "p-value — never a bare directional claim. **This is a research-"
        "screening tool, not a trading system.**\n\n"
        "**On the LLM layer (Layer 2):** the economic-mechanism explanations and "
        "plausibility flags are a **heuristic filter, not validation**. A "
        "language model can — and sometimes does — generate a confident, fluent "
        "mechanism for a relationship that is actually **spurious**. A "
        "'plausible' flag never upgrades a finding to causal; it only means "
        "'worth a closer human look'.",
        icon="⚠️",
    )


def main() -> None:
    st.set_page_config(
        page_title="Cross-Asset Causal Discovery Engine",
        page_icon="🕸️",
        layout="wide",
    )
    st.title("🕸️ Cross-Asset Causal Discovery Engine")
    st.caption(
        "Statistical causal discovery (Layer 1) + LLM plausibility / mechanism "
        "explanation (Layer 2). A thin client over the FastAPI service."
    )

    client = CausalAPIClient()
    demo_mode = is_demo_mode(client)
    if demo_mode:
        render_demo_banner()
    render_disclaimer()

    filters = render_sidebar(client, demo_mode)

    run_id = st.session_state.get("run_id")
    if not run_id:
        st.info(
            "👈 Load an existing `run_id` (or start a new analysis) from the "
            "sidebar to begin. Browsing an existing run is instant; a fresh run "
            "re-executes the multi-year pipeline and takes minutes."
        )
        return

    graph_tab, regime_tab, explore_tab, cards_tab, usecase_tab, events_tab = st.tabs(
        ["🕸️ Causal graph", "📈 Regime timeline", "🔎 Explore",
         "🧠 Hypothesis cards", "💼 Business use cases", "🚨 Events"]
    )
    with graph_tab:
        view_causal_graph(client, run_id, filters)
    with regime_tab:
        view_regime_timeline(client, run_id)
    with explore_tab:
        view_explore(client, run_id, filters)
    with cards_tab:
        view_hypothesis_cards(client, run_id, filters)
    with usecase_tab:
        view_business_use_cases(client, run_id)
    with events_tab:
        view_events(client, demo_mode)


# ``streamlit run dashboard/app.py`` executes this module as __main__.
if __name__ == "__main__":
    main()
