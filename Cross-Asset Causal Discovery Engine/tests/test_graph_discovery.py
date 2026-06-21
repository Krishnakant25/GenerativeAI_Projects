"""Tests for PC-based causal-graph discovery.

We validate the two behaviours that make PC an honest upgrade over a
correlation matrix, on *constructed* data where the ground-truth graph is
known (so the test is deterministic and needs no market data):

  1. A pure chain X -> Y -> Z is Markov-equivalent to its reverses, so its
     CPDAG leaves both edges UNDIRECTED. The engine must surface that as
     ``edge_type == "undirected"`` rather than inventing arrows, and must NOT
     connect X and Z directly (X _||_ Z given Y).

  2. A collider X -> Y <- Z (with X _||_ Z marginally) is a v-structure PC can
     orient from data alone: both edges point INTO Y, directed, with
     ``orientation_source == "pc"``. Again no direct X-Z edge.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import polars as pl

from causal.graph_discovery import discover_causal_graph


def _edge_between(graph: nx.DiGraph, u: str, v: str) -> dict | None:
    """Return the edge-data dict for u-v in either direction, or None."""
    if graph.has_edge(u, v):
        return graph.get_edge_data(u, v)
    if graph.has_edge(v, u):
        return graph.get_edge_data(v, u)
    return None


def _panel(columns: dict[str, np.ndarray]) -> pl.DataFrame:
    n = len(next(iter(columns.values())))
    return pl.DataFrame({"date": pl.Series("date", range(n)), **columns})


def test_chain_edges_left_undirected():
    """X -> Y -> Z recovers an undirected skeleton with no X-Z shortcut."""
    rng = np.random.default_rng(3)
    n = 5000
    x = rng.standard_normal(n)
    y = 0.8 * x + 0.3 * rng.standard_normal(n)
    z = 0.8 * y + 0.3 * rng.standard_normal(n)
    panel = _panel({"X": x, "Y": y, "Z": z})

    graph = discover_causal_graph(panel, alpha=0.05)

    xy = _edge_between(graph, "X", "Y")
    yz = _edge_between(graph, "Y", "Z")
    xz = _edge_between(graph, "X", "Z")

    # Skeleton: X-Y and Y-Z present, X-Z absent (screened off by Y).
    assert xy is not None, "expected an X-Y edge"
    assert yz is not None, "expected a Y-Z edge"
    assert xz is None, "X and Z must not be directly connected"

    # A Markov-equivalent chain is unorientable: both edges stay undirected.
    assert xy["edge_type"] == "undirected"
    assert yz["edge_type"] == "undirected"


def test_collider_is_oriented_by_pc():
    """X -> Y <- Z is a v-structure PC orients into Y from data alone."""
    rng = np.random.default_rng(5)
    n = 5000
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = 0.8 * x + 0.8 * z + 0.3 * rng.standard_normal(n)
    panel = _panel({"X": x, "Y": y, "Z": z})

    graph = discover_causal_graph(panel, alpha=0.05)

    # Both parents point INTO the collider Y, oriented by PC (not Granger).
    assert graph.has_edge("X", "Y"), "expected X -> Y"
    assert graph.has_edge("Z", "Y"), "expected Z -> Y"
    assert graph["X"]["Y"]["edge_type"] == "directed"
    assert graph["Z"]["Y"]["edge_type"] == "directed"
    assert graph["X"]["Y"]["orientation_source"] == "pc"
    assert graph["Z"]["Y"]["orientation_source"] == "pc"

    # The independent parents are not directly connected.
    assert _edge_between(graph, "X", "Z") is None


def test_every_node_present_even_if_isolated():
    """All assets appear as nodes, so an asset with no edges isn't dropped."""
    rng = np.random.default_rng(9)
    n = 1500
    panel = _panel(
        {
            "A": rng.standard_normal(n),
            "B": rng.standard_normal(n),
            "C": rng.standard_normal(n),
        }
    )
    graph = discover_causal_graph(panel, alpha=0.05)
    assert set(graph.nodes) == {"A", "B", "C"}
