"""SQLite persistence for analysis runs, causal candidates, and (Phase 2)
hypothesis cards.

The full schema — including the ``hypothesis_cards`` table — is created in
Phase 1. The ``hypothesis_cards`` table stays EMPTY until Phase 2 adds the LLM
layer; its read/write helpers exist now only so the storage contract is stable.

Design notes:
  - One ``analysis_runs`` row per end-to-end run of the engine. Every candidate
    is tagged with its run_id so results are reproducible and comparable across
    runs (e.g. regime-change early warning needs run-over-run diffing).
  - Statistical fields mirror the ``CausalCandidate`` Pydantic model. Per the
    hard rules, no causal edge is ever stored without its corrected p-value.
  - The discovered PC graph is **not** a separate table: it is exactly the
    subset of candidates PC kept, flagged via ``in_graph`` and annotated with
    ``edge_type`` / ``orientation_source``. One directed pair = one row, so the
    graph and the candidate list can never disagree about a relationship.
  - ``regime_periods`` and ``asset_universe`` are stored as JSON strings
    (SQLite has no array type).
  - Every statement is parameterized — no string interpolation of values.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from causal.models import (
    AnalysisRun,
    CausalCandidate,
    CausalGraph,
    GraphEdge,
    PairRegimes,
    RegimeFlipEvent,
    RegimePeriod,
)
from config import DB_PATH as DEFAULT_DB_PATH
from config import (
    DEFAULT_ALPHA,
    MONITOR_CONFIRMATION_RUNS,
)
from llm.models import HypothesisCard

SCHEMA = """
-- One row per end-to-end analysis run.
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,              -- ISO-8601 UTC
    start_date      TEXT NOT NULL,              -- data window start
    end_date        TEXT NOT NULL,              -- data window end
    asset_universe  TEXT NOT NULL,              -- JSON list of tickers analysed
    max_lag         INTEGER NOT NULL,           -- max lag tested (days)
    correction_method TEXT NOT NULL,            -- 'fdr_bh' | 'bonferroni'
    alpha           REAL NOT NULL,              -- significance threshold
    notes           TEXT
);

-- One row per directional candidate relationship (asset_a -> asset_b).
-- The PC-discovered graph is the subset where in_graph = 1.
CREATE TABLE IF NOT EXISTS causal_candidates (
    candidate_id        TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL,
    asset_a             TEXT NOT NULL,          -- driver (precedes)
    asset_b             TEXT NOT NULL,          -- affected (follows)
    direction           TEXT NOT NULL,          -- 'a_causes_b'
    lag                 INTEGER NOT NULL,       -- lag in days
    granger_p_value     REAL NOT NULL,          -- raw p-value
    corrected_p_value   REAL NOT NULL,          -- after FDR/Bonferroni
    correlation_strength REAL,                  -- peak time-lagged xcorr
    regime_periods      TEXT,                   -- JSON list of {start,end,active}
    statistical_confidence REAL NOT NULL,       -- 0-1 derived confidence
    is_significant      INTEGER NOT NULL,       -- 1 if passes corrected alpha
    in_graph            INTEGER NOT NULL DEFAULT 0,  -- 1 if PC kept this edge
    edge_type           TEXT,                   -- 'directed'|'undirected'|'bidirected'
    orientation_source  TEXT,                   -- 'pc'|'granger'|'none'
    FOREIGN KEY (run_id) REFERENCES analysis_runs (run_id)
);

CREATE INDEX IF NOT EXISTS idx_candidates_run ON causal_candidates (run_id);
CREATE INDEX IF NOT EXISTS idx_candidates_pair
    ON causal_candidates (asset_a, asset_b);

-- LAYER 2. Wraps a candidate with the LLM-generated economic mechanism +
-- plausibility flag (see llm/models.py). The underlying statistic is NOT copied
-- here: it lives once in causal_candidates and is JOINed back on read, so a card
-- can never disagree with — or silently mutate — the corrected p-value it
-- explains. One card per candidate per validation pass.
CREATE TABLE IF NOT EXISTS hypothesis_cards (
    card_id                 TEXT PRIMARY KEY,
    candidate_id            TEXT NOT NULL,
    asset_a                 TEXT NOT NULL,      -- denormalised for cheap queries
    asset_b                 TEXT NOT NULL,
    mechanism_explanation   TEXT NOT NULL,
    mechanism_channel       TEXT,              -- named textbook channel, or NULL
    plausibility_flag       TEXT NOT NULL,      -- PlausibilityFlag enum value
    llm_confidence          REAL NOT NULL,      -- LLM's own confidence 0-1
    caveats                 TEXT NOT NULL,      -- JSON list of strings
    addresses_pc_rejection  INTEGER NOT NULL,   -- 1 if it engaged in_graph=false
    model_name              TEXT NOT NULL,
    raw_response            TEXT,               -- raw model output (debug/parse fails)
    created_at              TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES causal_candidates (candidate_id)
);

CREATE INDEX IF NOT EXISTS idx_cards_candidate
    ON hypothesis_cards (candidate_id);

-- PHASE 3. One row per detected regime-status flip for a pair between two runs.
-- A flip carries the statistic of its NEW-run candidate (hard rule: no flip
-- without its corrected p-value). It starts 'pending' and is only 'confirmed'
-- once its new status survives MONITOR_CONFIRMATION_RUNS consecutive monitor
-- runs, so a yfinance data-revision blip on the latest bars is never presented
-- as a confident finding. A flip that snaps back first becomes 'reverted'.
CREATE TABLE IF NOT EXISTS regime_flip_events (
    event_id            TEXT PRIMARY KEY,
    asset_a             TEXT NOT NULL,
    asset_b             TEXT NOT NULL,
    prior_run_id        TEXT NOT NULL,      -- run it flipped FROM
    new_run_id          TEXT NOT NULL,      -- run first detected in
    old_active          INTEGER NOT NULL,   -- coupling status before
    new_active          INTEGER NOT NULL,   -- coupling status after
    old_mean_correlation REAL,
    new_mean_correlation REAL,
    corrected_p_value   REAL NOT NULL,      -- new-run candidate's corrected p
    lag                 INTEGER NOT NULL,
    correlation_strength REAL,
    status              TEXT NOT NULL,      -- 'pending'|'confirmed'|'reverted'
    consecutive_confirmations INTEGER NOT NULL,
    detected_at         TEXT NOT NULL,      -- ISO-8601 UTC
    last_seen_run_id    TEXT,               -- latest run the new status held in
    updated_at          TEXT NOT NULL,
    FOREIGN KEY (new_run_id) REFERENCES analysis_runs (run_id),
    FOREIGN KEY (prior_run_id) REFERENCES analysis_runs (run_id)
);

CREATE INDEX IF NOT EXISTS idx_flips_pair
    ON regime_flip_events (asset_a, asset_b);
CREATE INDEX IF NOT EXISTS idx_flips_status ON regime_flip_events (status);
"""

# Columns that uniquely identify the *current* hypothesis_cards schema. An older
# Phase-1 stub shipped a different layout (mechanism/plausibility/confidence).
# The table is always empty until Layer 2 first runs, so a stale layout is safe
# to drop and recreate rather than migrate row-by-row.
_HYPOTHESIS_CARDS_REQUIRED_COLS = frozenset(
    {"mechanism_explanation", "plausibility_flag", "llm_confidence", "caveats"}
)


def _drop_stale_hypothesis_cards(conn: sqlite3.Connection) -> None:
    """Drop ``hypothesis_cards`` if it exists with an out-of-date column set.

    Safe because the table is empty until Layer 2 populates it. Logs nothing —
    ``init_db`` recreates the table immediately afterwards from ``SCHEMA``.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(hypothesis_cards)")}
    if cols and not _HYPOTHESIS_CARDS_REQUIRED_COLS.issubset(cols):
        # Old stub schema (or partial). Empty by contract — drop and recreate.
        n = conn.execute("SELECT COUNT(*) FROM hypothesis_cards").fetchone()[0]
        if n == 0:
            conn.execute("DROP TABLE hypothesis_cards")


# --------------------------------------------------------------------------
# Connection / schema
# --------------------------------------------------------------------------


def get_connection(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enforced and Row access."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> Path:
    """Create all tables if they do not exist. Returns the DB path.

    Idempotent — safe to call on every startup.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        _drop_stale_hypothesis_cards(conn)
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return db_path


# --------------------------------------------------------------------------
# (De)serialization helpers
# --------------------------------------------------------------------------


def _regimes_to_json(periods: list[RegimePeriod]) -> str:
    return json.dumps([p.model_dump(mode="json") for p in periods])


def _regimes_from_json(blob: str | None) -> list[RegimePeriod]:
    if not blob:
        return []
    return [RegimePeriod(**d) for d in json.loads(blob)]


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------


def insert_run(conn: sqlite3.Connection, run: AnalysisRun) -> None:
    """Insert one analysis-run row (parameterized)."""
    conn.execute(
        """
        INSERT INTO analysis_runs
            (run_id, created_at, start_date, end_date, asset_universe,
             max_lag, correction_method, alpha, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run.run_id,
            run.created_at,
            run.start_date,
            run.end_date,
            json.dumps(run.asset_universe),
            run.max_lag,
            run.correction_method,
            run.alpha,
            run.notes,
        ),
    )


def insert_candidates(
    conn: sqlite3.Connection,
    candidates: list[CausalCandidate],
    graph_meta: dict[str, tuple[str, str]] | None = None,
) -> None:
    """Insert candidate rows (parameterized, batched).

    ``graph_meta`` maps ``candidate_id -> (edge_type, orientation_source)`` for
    the candidates PC kept as graph edges; everything else is stored with
    ``in_graph = 0``.
    """
    graph_meta = graph_meta or {}
    rows = []
    for c in candidates:
        edge = graph_meta.get(c.candidate_id)
        in_graph = 1 if edge else 0
        edge_type, orientation_source = edge if edge else (None, None)
        rows.append(
            (
                c.candidate_id,
                c.run_id,
                c.asset_a,
                c.asset_b,
                c.direction,
                c.lag,
                c.granger_p_value,
                c.corrected_p_value,
                c.correlation_strength,
                _regimes_to_json(c.regime_periods),
                c.statistical_confidence,
                1 if c.is_significant else 0,
                in_graph,
                edge_type,
                orientation_source,
            )
        )
    conn.executemany(
        """
        INSERT INTO causal_candidates
            (candidate_id, run_id, asset_a, asset_b, direction, lag,
             granger_p_value, corrected_p_value, correlation_strength,
             regime_periods, statistical_confidence, is_significant,
             in_graph, edge_type, orientation_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def persist_run(
    run: AnalysisRun,
    candidates: list[CausalCandidate],
    graph_meta: dict[str, tuple[str, str]] | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    """Write a run and all its candidates in a single transaction.

    Returns the run_id. Rolls back atomically if any insert fails.
    """
    conn = get_connection(db_path)
    try:
        with conn:  # commits on success, rolls back on exception
            insert_run(conn, run)
            insert_candidates(conn, candidates, graph_meta)
    finally:
        conn.close()
    return run.run_id


def insert_hypothesis_card(
    card: HypothesisCard,
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> str:
    """Insert one Layer-2 hypothesis card (parameterized). Returns the card_id.

    The underlying statistic is NOT written here — only the LLM narrative and
    flags. ``card.candidate`` already lives in ``causal_candidates`` and is
    JOINed back on read, so the corrected p-value passes through untouched.

    ``card.card_id`` / ``card.created_at`` are used if set, else generated, and
    the stored values are returned on the card-id.
    """
    card_id = card.card_id or uuid.uuid4().hex
    created_at = card.created_at or datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO hypothesis_cards
                    (card_id, candidate_id, asset_a, asset_b,
                     mechanism_explanation, mechanism_channel, plausibility_flag,
                     llm_confidence, caveats, addresses_pc_rejection,
                     model_name, raw_response, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    card_id,
                    card.candidate_id,
                    card.asset_a,
                    card.asset_b,
                    card.mechanism_explanation,
                    card.mechanism_channel,
                    # ``use_enum_values`` means plausibility_flag is already a str.
                    card.plausibility_flag,
                    card.llm_confidence,
                    json.dumps(card.caveats),
                    1 if card.addresses_pc_rejection else 0,
                    card.model_name,
                    card.raw_response,
                    created_at,
                ),
            )
    finally:
        conn.close()
    return card_id


def replace_hypothesis_cards(
    run_id: str,
    cards: list[HypothesisCard],
    *,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> int:
    """Persist a full validation pass for one run atomically.

    Deletes any prior cards for the run's candidates and inserts the new batch
    in a single transaction, so re-validating a run replaces rather than
    duplicates its cards. Returns the number of cards written.
    """
    conn = get_connection(db_path)
    try:
        with conn:
            # Clear out cards belonging to this run's candidates.
            conn.execute(
                """
                DELETE FROM hypothesis_cards
                WHERE candidate_id IN (
                    SELECT candidate_id FROM causal_candidates WHERE run_id = ?
                )
                """,
                (run_id,),
            )
            for card in cards:
                card_id = card.card_id or uuid.uuid4().hex
                created_at = card.created_at or datetime.now(
                    timezone.utc
                ).isoformat()
                conn.execute(
                    """
                    INSERT INTO hypothesis_cards
                        (card_id, candidate_id, asset_a, asset_b,
                         mechanism_explanation, mechanism_channel,
                         plausibility_flag, llm_confidence, caveats,
                         addresses_pc_rejection, model_name, raw_response,
                         created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card_id,
                        card.candidate_id,
                        card.asset_a,
                        card.asset_b,
                        card.mechanism_explanation,
                        card.mechanism_channel,
                        card.plausibility_flag,
                        card.llm_confidence,
                        json.dumps(card.caveats),
                        1 if card.addresses_pc_rejection else 0,
                        card.model_name,
                        card.raw_response,
                        created_at,
                    ),
                )
    finally:
        conn.close()
    return len(cards)


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def _row_to_candidate(row: sqlite3.Row) -> CausalCandidate:
    return CausalCandidate(
        candidate_id=row["candidate_id"],
        run_id=row["run_id"],
        asset_a=row["asset_a"],
        asset_b=row["asset_b"],
        direction=row["direction"],
        lag=row["lag"],
        granger_p_value=row["granger_p_value"],
        corrected_p_value=row["corrected_p_value"],
        correlation_strength=row["correlation_strength"],
        regime_periods=_regimes_from_json(row["regime_periods"]),
        statistical_confidence=row["statistical_confidence"],
        is_significant=bool(row["is_significant"]),
    )


def _row_to_run(row: sqlite3.Row) -> AnalysisRun:
    return AnalysisRun(
        run_id=row["run_id"],
        created_at=row["created_at"],
        start_date=row["start_date"],
        end_date=row["end_date"],
        asset_universe=json.loads(row["asset_universe"]),
        max_lag=row["max_lag"],
        correction_method=row["correction_method"],
        alpha=row["alpha"],
        notes=row["notes"],
    )


def load_run(
    run_id: str, db_path: Path | str = DEFAULT_DB_PATH
) -> AnalysisRun | None:
    """Load a single run's metadata, or None if the run_id is unknown."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM analysis_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()
    return _row_to_run(row) if row is not None else None


def list_runs(db_path: Path | str = DEFAULT_DB_PATH) -> list[AnalysisRun]:
    """All analysis runs in chronological order (oldest first).

    Phase 3 uses this to pick the two most recent runs to diff for regime flips.
    Ordering is by ``created_at`` (the run_id also embeds a UTC timestamp, but
    created_at is the authoritative field).
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM analysis_runs ORDER BY created_at ASC, run_id ASC"
        ).fetchall()
    finally:
        conn.close()
    return [_row_to_run(r) for r in rows]


def load_candidates(
    run_id: str,
    significant_only: bool = False,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[CausalCandidate]:
    """Load a run's candidates, ordered by corrected p-value (most significant
    first). Set ``significant_only`` to return only edges that passed alpha."""
    query = "SELECT * FROM causal_candidates WHERE run_id = ?"
    params: tuple = (run_id,)
    if significant_only:
        query += " AND is_significant = 1"
    query += " ORDER BY corrected_p_value ASC"

    conn = get_connection(db_path)
    try:
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()
    return [_row_to_candidate(r) for r in rows]


def load_graph(
    run_id: str, db_path: Path | str = DEFAULT_DB_PATH
) -> CausalGraph | None:
    """Reconstruct the discovered causal graph for a run as a node-link object.

    Nodes are the run's full asset universe; edges are the candidates PC kept
    (``in_graph = 1``), each carrying the statistic that justifies it. Returns
    None if the run_id is unknown.
    """
    run = load_run(run_id, db_path)
    if run is None:
        return None

    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT asset_a, asset_b, edge_type, orientation_source,
                   corrected_p_value, lag, correlation_strength
            FROM causal_candidates
            WHERE run_id = ? AND in_graph = 1
            ORDER BY corrected_p_value ASC
            """,
            (run_id,),
        ).fetchall()
    finally:
        conn.close()

    edges = [
        GraphEdge(
            source=r["asset_a"],
            target=r["asset_b"],
            edge_type=r["edge_type"],
            orientation_source=r["orientation_source"],
            corrected_p_value=r["corrected_p_value"],
            lag=r["lag"],
            correlation_strength=r["correlation_strength"],
        )
        for r in rows
    ]
    return CausalGraph(run_id=run_id, nodes=run.asset_universe, edges=edges)


def load_regimes(
    run_id: str, db_path: Path | str = DEFAULT_DB_PATH
) -> list[PairRegimes]:
    """Load time-bound regime windows for every pair that has them (the
    significant candidates regime detection was run on)."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            """
            SELECT asset_a, asset_b, regime_periods
            FROM causal_candidates
            WHERE run_id = ? AND regime_periods IS NOT NULL
              AND regime_periods != '[]'
            ORDER BY asset_a, asset_b
            """,
            (run_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        PairRegimes(
            asset_a=r["asset_a"],
            asset_b=r["asset_b"],
            regime_periods=_regimes_from_json(r["regime_periods"]),
        )
        for r in rows
    ]


# --------------------------------------------------------------------------
# Phase 3 — regime-flip detection & confirmation tracking
# --------------------------------------------------------------------------


def _current_regime_status(
    periods: list[RegimePeriod],
) -> tuple[bool, float | None] | None:
    """The pair's *current* coupling status = the most recent regime window's
    ``active`` flag (and its mean correlation). ``None`` if the pair has no
    windows (it was not significant, so regimes were never fit).

    Note: ``active`` is assigned per-run by the HMM stage from the higher
    mean-|correlation| state (``argmax`` over fitted means), NOT by a raw HMM
    state index — so comparing this flag across two independent fits is
    label-switching-safe by construction (see causal/regime_detection.py).
    """
    if not periods:
        return None
    last = periods[-1]
    return bool(last.active), last.mean_correlation


def _candidate_status_map(
    run_id: str,
    *,
    alpha: float,
    db_path: Path | str,
) -> dict[tuple[str, str], tuple[bool, float | None, CausalCandidate]]:
    """Map each *significant, alpha-passing* pair of a run to its current regime
    status and underlying candidate. This is the alert-fatigue gate: pairs that
    were never significant after correction are excluded outright."""
    out: dict[tuple[str, str], tuple[bool, float | None, CausalCandidate]] = {}
    for cand in load_candidates(run_id, significant_only=True, db_path=db_path):
        if cand.corrected_p_value > alpha:
            continue  # explicit alpha gate (belt-and-suspenders over is_significant)
        status = _current_regime_status(cand.regime_periods)
        if status is None:
            continue
        active, mean_corr = status
        out[(cand.asset_a, cand.asset_b)] = (active, mean_corr, cand)
    return out


def diff_regimes(
    run_a_id: str,
    run_b_id: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[RegimeFlipEvent]:
    """Pure diff: pairs whose *current* regime status changed from run A to B.

    Only pairs that are significant (alpha-gated) in the **new** run (B) and have
    a comparable current status in **both** runs are considered — a pair that
    only became significant in B is a new relationship, not a flip. Each returned
    event carries B's statistic and defaults to ``pending`` (the caller applies
    confirmation tracking). This function persists nothing.
    """
    old_map = _candidate_status_map(run_a_id, alpha=1.0, db_path=db_path)
    new_map = _candidate_status_map(run_b_id, alpha=alpha, db_path=db_path)
    now = datetime.now(timezone.utc).isoformat()

    events: list[RegimeFlipEvent] = []
    for pair, (new_active, new_corr, cand) in new_map.items():
        old = old_map.get(pair)
        if old is None:
            continue  # no comparable prior status -> not a flip
        old_active, old_corr, _ = old
        if old_active == new_active:
            continue
        events.append(
            RegimeFlipEvent(
                asset_a=pair[0],
                asset_b=pair[1],
                prior_run_id=run_a_id,
                new_run_id=run_b_id,
                old_active=old_active,
                new_active=new_active,
                old_mean_correlation=old_corr,
                new_mean_correlation=new_corr,
                corrected_p_value=cand.corrected_p_value,
                lag=cand.lag,
                correlation_strength=cand.correlation_strength,
                status="pending",
                consecutive_confirmations=1,
                detected_at=now,
                last_seen_run_id=run_b_id,
            )
        )
    return events


def _row_to_flip(row: sqlite3.Row) -> RegimeFlipEvent:
    return RegimeFlipEvent(
        asset_a=row["asset_a"],
        asset_b=row["asset_b"],
        prior_run_id=row["prior_run_id"],
        new_run_id=row["new_run_id"],
        old_active=bool(row["old_active"]),
        new_active=bool(row["new_active"]),
        old_mean_correlation=row["old_mean_correlation"],
        new_mean_correlation=row["new_mean_correlation"],
        corrected_p_value=row["corrected_p_value"],
        lag=row["lag"],
        correlation_strength=row["correlation_strength"],
        status=row["status"],
        consecutive_confirmations=row["consecutive_confirmations"],
        detected_at=row["detected_at"],
        last_seen_run_id=row["last_seen_run_id"],
    )


def record_and_update_flips(
    prior_run_id: str,
    new_run_id: str,
    *,
    alpha: float = DEFAULT_ALPHA,
    n_confirm: int = MONITOR_CONFIRMATION_RUNS,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[RegimeFlipEvent]:
    """Advance flip tracking by one monitor cycle and return the events touched.

    Two stages, in order:
      1. **Confirm/expire** every still-open ('pending') flip against the new
         run's current status: if the new status held, increment its consecutive
         count (→ 'confirmed' at ``n_confirm``); if it snapped back, mark it
         'reverted'; if the pair is no longer significant, also 'reverted'.
      2. **Detect** brand-new flips between ``prior_run_id`` and ``new_run_id``
         (``diff_regimes``) and insert any that aren't already being tracked.

    Idempotent within a run: re-running for the same ``new_run_id`` will not
    double-increment (guarded by ``last_seen_run_id``).
    """
    new_map = _candidate_status_map(new_run_id, alpha=alpha, db_path=db_path)
    now = datetime.now(timezone.utc).isoformat()
    touched: list[RegimeFlipEvent] = []

    conn = get_connection(db_path)
    try:
        with conn:
            # --- Stage 1: confirm or expire open (pending) flips --------------
            open_rows = conn.execute(
                "SELECT * FROM regime_flip_events WHERE status = 'pending'"
            ).fetchall()
            for row in open_rows:
                flip = _row_to_flip(row)
                pair = (flip.asset_a, flip.asset_b)
                cur = new_map.get(pair)
                new_status: str = flip.status
                consecutive = flip.consecutive_confirmations
                last_seen = flip.last_seen_run_id

                if cur is None:
                    # No longer significant / observable -> cannot confirm.
                    new_status = "reverted"
                else:
                    cur_active = cur[0]
                    if cur_active == flip.new_active:
                        # The new status held. Count this run once.
                        if flip.last_seen_run_id != new_run_id:
                            consecutive += 1
                            last_seen = new_run_id
                        if consecutive >= n_confirm:
                            new_status = "confirmed"
                    elif cur_active == flip.old_active:
                        new_status = "reverted"  # snapped back

                conn.execute(
                    """
                    UPDATE regime_flip_events
                       SET status = ?, consecutive_confirmations = ?,
                           last_seen_run_id = ?, updated_at = ?
                     WHERE event_id = ?
                    """,
                    (new_status, consecutive, last_seen, now, row["event_id"]),
                )
                touched.append(
                    _row_to_flip(
                        conn.execute(
                            "SELECT * FROM regime_flip_events WHERE event_id = ?",
                            (row["event_id"],),
                        ).fetchone()
                    )
                )

            # --- Stage 2: detect & insert brand-new flips --------------------
            for ev in diff_regimes(
                prior_run_id, new_run_id, alpha=alpha, db_path=db_path
            ):
                existing = conn.execute(
                    """
                    SELECT 1 FROM regime_flip_events
                     WHERE asset_a = ? AND asset_b = ?
                       AND old_active = ? AND new_active = ?
                       AND status IN ('pending', 'confirmed')
                    """,
                    (ev.asset_a, ev.asset_b, int(ev.old_active), int(ev.new_active)),
                ).fetchone()
                if existing:
                    continue  # already tracking this ongoing flip
                event_id = uuid.uuid4().hex
                conn.execute(
                    """
                    INSERT INTO regime_flip_events
                        (event_id, asset_a, asset_b, prior_run_id, new_run_id,
                         old_active, new_active, old_mean_correlation,
                         new_mean_correlation, corrected_p_value, lag,
                         correlation_strength, status, consecutive_confirmations,
                         detected_at, last_seen_run_id, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        ev.asset_a,
                        ev.asset_b,
                        ev.prior_run_id,
                        ev.new_run_id,
                        int(ev.old_active),
                        int(ev.new_active),
                        ev.old_mean_correlation,
                        ev.new_mean_correlation,
                        ev.corrected_p_value,
                        ev.lag,
                        ev.correlation_strength,
                        ev.status,
                        ev.consecutive_confirmations,
                        ev.detected_at,
                        ev.last_seen_run_id,
                        now,
                    ),
                )
                touched.append(
                    _row_to_flip(
                        conn.execute(
                            "SELECT * FROM regime_flip_events WHERE event_id = ?",
                            (event_id,),
                        ).fetchone()
                    )
                )
    finally:
        conn.close()
    return touched


def load_flip_events(
    *,
    asset_a: str | None = None,
    asset_b: str | None = None,
    status: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[RegimeFlipEvent]:
    """Load regime-flip events, most recently detected first. Optionally filter
    by pair and/or status ('pending'|'confirmed'|'reverted')."""
    clauses: list[str] = []
    params: list = []
    if asset_a is not None:
        clauses.append("asset_a = ?")
        params.append(asset_a)
    if asset_b is not None:
        clauses.append("asset_b = ?")
        params.append(asset_b)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)

    query = "SELECT * FROM regime_flip_events"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY detected_at DESC, asset_a ASC, asset_b ASC"

    conn = get_connection(db_path)
    try:
        rows = conn.execute(query, tuple(params)).fetchall()
    finally:
        conn.close()
    return [_row_to_flip(r) for r in rows]


def _row_to_card(row: sqlite3.Row) -> HypothesisCard:
    """Rebuild a HypothesisCard from a hypothesis_cards ⋈ causal_candidates row.

    The embedded statistic is reconstructed straight from causal_candidates, so
    the card always carries the exact corrected p-value Layer 1 recorded.
    """
    candidate = CausalCandidate(
        candidate_id=row["candidate_id"],
        run_id=row["run_id"],
        asset_a=row["asset_a"],
        asset_b=row["asset_b"],
        direction=row["direction"],
        lag=row["lag"],
        granger_p_value=row["granger_p_value"],
        corrected_p_value=row["corrected_p_value"],
        correlation_strength=row["correlation_strength"],
        regime_periods=_regimes_from_json(row["regime_periods"]),
        statistical_confidence=row["statistical_confidence"],
        is_significant=bool(row["is_significant"]),
    )
    return HypothesisCard(
        card_id=row["card_id"],
        candidate=candidate,
        in_graph=bool(row["in_graph"]),
        mechanism_explanation=row["mechanism_explanation"],
        mechanism_channel=row["mechanism_channel"],
        plausibility_flag=row["plausibility_flag"],
        llm_confidence=row["llm_confidence"],
        caveats=json.loads(row["caveats"]) if row["caveats"] else [],
        addresses_pc_rejection=bool(row["addresses_pc_rejection"]),
        model_name=row["model_name"],
        raw_response=row["raw_response"],
        created_at=row["created_at"],
    )


_CARD_JOIN_SELECT = """
    SELECT
        hc.card_id, hc.mechanism_explanation, hc.mechanism_channel,
        hc.plausibility_flag, hc.llm_confidence, hc.caveats,
        hc.addresses_pc_rejection, hc.model_name, hc.raw_response, hc.created_at,
        cc.candidate_id, cc.run_id, cc.asset_a, cc.asset_b, cc.direction, cc.lag,
        cc.granger_p_value, cc.corrected_p_value, cc.correlation_strength,
        cc.regime_periods, cc.statistical_confidence, cc.is_significant,
        cc.in_graph
    FROM hypothesis_cards hc
    JOIN causal_candidates cc ON cc.candidate_id = hc.candidate_id
"""


def load_hypothesis_cards(
    run_id: str | None = None,
    candidate_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[HypothesisCard]:
    """Load Layer-2 hypothesis cards, each carrying its full underlying
    statistic (rebuilt via JOIN). Sorted by LLM confidence, most confident
    first. Filter by ``run_id`` and/or ``candidate_id``."""
    clauses = []
    params: list = []
    if run_id is not None:
        clauses.append("cc.run_id = ?")
        params.append(run_id)
    if candidate_id is not None:
        clauses.append("hc.candidate_id = ?")
        params.append(candidate_id)

    query = _CARD_JOIN_SELECT
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY hc.llm_confidence DESC, cc.corrected_p_value ASC"

    conn = get_connection(db_path)
    try:
        rows = conn.execute(query, tuple(params)).fetchall()
    finally:
        conn.close()
    return [_row_to_card(r) for r in rows]


if __name__ == "__main__":
    path = init_db()
    print(f"Initialised SQLite schema at: {path}")
