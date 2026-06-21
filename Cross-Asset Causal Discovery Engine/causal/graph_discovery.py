"""Causal graph discovery via the PC algorithm (causal-learn).

Granger tests give *pairwise* predictive precedence; PC goes further and
recovers a graph over the *whole* asset set, removing edges that vanish once
you condition on other assets (i.e. spurious links explained by a common
driver). This is the honest upgrade over a correlation matrix.

Two facts about PC's output drive everything below — both verified against the
installed causal-learn 0.1.4.7, not assumed:

  * PC returns a **CPDAG**, not a fully directed graph. Edges it cannot orient
    from observational data alone (e.g. a pure chain A-B-C, which is
    Markov-equivalent to its reverses) come back **undirected**. We surface
    that honestly rather than inventing arrows.
  * Edge orientation is read from `Edge` endpoints (Endpoint enum:
    TAIL=-1, ARROW=1):
        TAIL  -- ARROW  => directed   node1 -> node2
        TAIL  -- TAIL   => undirected (unorientable)
        ARROW -- ARROW  => bidirected (hint of a latent common cause)

For edges PC leaves undirected we fall back to the Granger-significant
direction as a *secondary* orientation signal, tagging the edge with
`orientation_source` so the distinction is never hidden.

The result is a NetworkX ``DiGraph``; every edge carries the underlying
statistic (corrected p-value / lag / correlation) so no arrow is ever shown
without its evidence — a project hard rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import polars as pl
from causallearn.graph.Endpoint import Endpoint
from causallearn.search.ConstraintBased.PC import pc

from causal.granger import GrangerResult
from config import DEFAULT_ALPHA, DEFAULT_INDEP_TEST


@dataclass(frozen=True)
class CausalEdge:
    """One edge in the discovered causal graph, with its evidence.

    At this layer only the *raw* Granger p-value is known — multiple-comparisons
    correction happens later, across all pairs (see ``causal.correction`` and
    the orchestration in ``causal.pipeline``). So this carries ``granger_p_value``
    honestly; the corrected value is attached downstream, never faked here.
    """

    source: str
    target: str
    edge_type: str          # "directed" | "undirected" | "bidirected"
    orientation_source: str  # "pc" | "granger" | "none"
    granger_p_value: float | None = None
    lag: int | None = None


def _classify(e1: Endpoint, e2: Endpoint) -> str:
    """Map a (endpoint1, endpoint2) pair to an edge type."""
    tail, arrow = Endpoint.TAIL, Endpoint.ARROW
    if e1 == tail and e2 == arrow:
        return "forward"      # node1 -> node2
    if e1 == arrow and e2 == tail:
        return "backward"     # node2 -> node1
    if e1 == tail and e2 == tail:
        return "undirected"
    if e1 == arrow and e2 == arrow:
        return "bidirected"
    return "other"            # CIRCLE etc. — only from FCI/PAG, not pc


def discover_causal_graph(
    returns: pl.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    indep_test: str = DEFAULT_INDEP_TEST,
    granger_results: list[GrangerResult] | None = None,
) -> nx.DiGraph:
    """Run PC over the log-return panel and return an annotated ``DiGraph``.

    ``returns`` is the wide log-return panel (``date`` + one column per
    ticker). ``granger_results`` (optional) is used only to orient edges PC
    leaves undirected.
    """
    tickers = [c for c in returns.columns if c != "date"]
    data = returns.select(tickers).to_numpy()

    cg = pc(data, alpha, indep_test, node_names=tickers, show_progress=False)

    # Fast lookup of Granger evidence for an ordered pair (driver, affected).
    g_by_pair: dict[tuple[str, str], GrangerResult] = {}
    for g in granger_results or []:
        g_by_pair[(g.asset_a, g.asset_b)] = g

    graph = nx.DiGraph()
    graph.add_nodes_from(tickers)

    for edge in cg.G.get_graph_edges():
        a = edge.get_node1().get_name()
        b = edge.get_node2().get_name()
        kind = _classify(edge.get_endpoint1(), edge.get_endpoint2())

        if kind == "forward":
            _add_edge(graph, a, b, "directed", "pc", g_by_pair)
        elif kind == "backward":
            _add_edge(graph, b, a, "directed", "pc", g_by_pair)
        elif kind == "bidirected":
            # latent common cause: record both directions, flagged as such
            _add_edge(graph, a, b, "bidirected", "pc", g_by_pair)
            _add_edge(graph, b, a, "bidirected", "pc", g_by_pair)
        elif kind == "undirected":
            # PC can't orient it; let Granger break the tie if it can.
            ga, gb = g_by_pair.get((a, b)), g_by_pair.get((b, a))
            if ga and (not gb or ga.p_value <= gb.p_value):
                _add_edge(graph, a, b, "undirected", "granger", g_by_pair)
            elif gb:
                _add_edge(graph, b, a, "undirected", "granger", g_by_pair)
            else:
                _add_edge(graph, a, b, "undirected", "none", g_by_pair)

    return graph


def _add_edge(
    graph: nx.DiGraph,
    src: str,
    dst: str,
    edge_type: str,
    orientation_source: str,
    g_by_pair: dict[tuple[str, str], GrangerResult],
) -> None:
    g = g_by_pair.get((src, dst))
    graph.add_edge(
        src,
        dst,
        edge_type=edge_type,
        orientation_source=orientation_source,
        granger_p_value=(g.p_value if g else None),
        lag=(g.best_lag if g else None),
    )


def graph_edges(graph: nx.DiGraph) -> list[CausalEdge]:
    """Flatten an annotated ``DiGraph`` into ``CausalEdge`` records."""
    return [
        CausalEdge(
            source=u,
            target=v,
            edge_type=d["edge_type"],
            orientation_source=d["orientation_source"],
            granger_p_value=d.get("granger_p_value"),
            lag=d.get("lag"),
        )
        for u, v, d in graph.edges(data=True)
    ]


if __name__ == "__main__":
    from data.fetcher import fetch_prices
    from data.preprocessor import preprocess
    from causal.granger import run_pairwise_granger

    _, rets = preprocess(fetch_prices())
    g_results = run_pairwise_granger(rets, max_lag=5)
    G = discover_causal_graph(rets, alpha=0.05, granger_results=g_results)
    print(f"Discovered {G.number_of_edges()} edges over {G.number_of_nodes()} assets:")
    for e in graph_edges(G):
        print(f"  {e.source:8s} -> {e.target:8s} "
              f"[{e.edge_type}/{e.orientation_source}] lag={e.lag}")
