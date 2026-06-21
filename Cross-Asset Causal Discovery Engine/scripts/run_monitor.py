"""Scheduled regime-flip monitor (Phase 3).

One monitor *cycle* = run the full Layer-1 pipeline against fresh data, persist
it as a new run, then diff the new run's per-pair regime status against the most
recent prior run and advance flip tracking (detect new flips, confirm/expire
existing ones). Every flip carries the statistic that justifies it, and no flip
is treated as a confident finding until it survives the confirmation window
(``config.MONITOR_CONFIRMATION_RUNS`` consecutive runs) — the same "don't trust
a lone extreme number" discipline the rest of the project follows.

SCHEDULING — why single-cycle + Windows Task Scheduler is the default:
  The recommended production setup is a **single cycle per invocation** driven by
  **Windows Task Scheduler** (e.g. daily after the US close). Rationale: no
  long-lived Python process to leak memory or die silently across a reboot, the
  OS owns the schedule and retry policy, and each cycle's log is a self-contained
  operational record. A ``--loop`` mode (sleep between cycles in-process) is
  provided for demos / machines without Task Scheduler, but is explicitly the
  fallback, not the recommendation. No cloud, no extra dependency — the loop is
  a plain ``time.sleep`` (default interval ``--interval 86400`` = one day).

  Why ``--loop`` is *demo-only*, not just "less robust": the confirmation window
  (``config.MONITOR_CONFIRMATION_RUNS``) only carries meaning when consecutive
  cycles are separated by REAL time at the data's own granularity — daily bars,
  so ~1 day apart. A flip is "confirmed" precisely because its new regime status
  survived several *independent, later* observations of fresh data. Spinning the
  loop fast (a cycle every few seconds) re-runs the SAME yfinance bars: the
  status trivially "holds" and "confirmed after N runs" becomes vacuously true —
  the exact data-revision guard the confirmation window exists to provide is
  defeated. So a short ``--interval`` is fine for *exercising the plumbing* in a
  demo, but a genuine confirmation requires real calendar time between cycles,
  which is why production should be Task Scheduler firing once per day.

Usage:
    python -m scripts.run_monitor                      # one cycle, latest data
    python -m scripts.run_monitor --end-date 2026-06-01  # snapshot at a date
    python -m scripts.run_monitor --loop --interval 86400  # daily, in-process
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from causal.models import RegimeFlipEvent
from causal.pipeline import InsufficientDataError, run_analysis
from config import asset_name
from data.fetcher import DataUnavailableError
from db import storage

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOG_PATH = RESULTS_DIR / "monitor.log"

logger = logging.getLogger("causal_engine.monitor")


def _configure_logging() -> None:
    """Timestamped logs to BOTH stdout and results/monitor.log (the operational
    record). Idempotent: safe to call once per process."""
    if logger.handlers:
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.INFO)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)


def _fmt_flip(ev: RegimeFlipEvent) -> str:
    return (
        f"{asset_name(ev.asset_a)} -> {asset_name(ev.asset_b)} "
        f"[{ev.direction}] status={ev.status} "
        f"({ev.consecutive_confirmations}/{config.MONITOR_CONFIRMATION_RUNS}) "
        f"corrected_p={ev.corrected_p_value:.2e} lag={ev.lag}d"
    )


def run_monitor_cycle(
    *,
    start_date: str = config.DEFAULT_START_DATE,
    end_date: str | None = config.DEFAULT_END_DATE,
    alpha: float = config.MONITOR_MIN_SIGNIFICANCE_ALPHA,
    n_confirm: int = config.MONITOR_CONFIRMATION_RUNS,
    db_path: Path | str = config.DB_PATH,
) -> list[RegimeFlipEvent]:
    """Run one monitor cycle. Returns the flip events touched this cycle.

    Persists a fresh run, then (if a prior run exists) advances flip tracking
    against it. Never raises on "no data to compare yet" — the first ever run
    simply records a baseline and logs that there was nothing to diff.
    """
    t0 = time.time()
    window = f"{start_date} .. {end_date or 'latest'}"
    logger.info("Monitor cycle start | window %s | alpha=%s n_confirm=%s",
                window, alpha, n_confirm)

    # The monitor is a standalone entry point (Task Scheduler / CLI), so it owns
    # its schema the same way the API does in its lifespan — ensure the tables
    # exist before the first persist. Idempotent and cheap; required when pointed
    # at a fresh --db path that no API/`python -m db.storage` has touched yet.
    storage.init_db(db_path)

    try:
        result = run_analysis(
            start_date=start_date,
            end_date=end_date,
            alpha=alpha,
            notes=config.MONITOR_RUN_NOTE,
        )
    except DataUnavailableError as exc:
        logger.error("Data provider returned nothing usable: %s", exc)
        raise
    except InsufficientDataError as exc:
        logger.error("Aligned panel too short to analyse: %s", exc)
        raise

    storage.persist_run(
        result.run, result.candidates, result.graph_meta, db_path=db_path
    )
    n_sig = sum(1 for c in result.candidates if c.is_significant)
    logger.info(
        "Persisted run %s | covered %s..%s | %d candidates (%d significant)",
        result.run.run_id, result.run.start_date, result.run.end_date,
        len(result.candidates), n_sig,
    )

    runs = storage.list_runs(db_path=db_path)
    if len(runs) < 2:
        logger.info(
            "Only one run on record — baseline established, nothing to diff yet. "
            "Run the monitor again (against later data) to detect flips."
        )
        logger.info("Monitor cycle done in %.1fs", time.time() - t0)
        return []

    prior, new = runs[-2], runs[-1]
    if new.run_id != result.run.run_id:  # defensive: ordering sanity
        logger.warning(
            "Newest run %s is not the one just persisted (%s); diffing newest two.",
            new.run_id, result.run.run_id,
        )
    logger.info("Diffing regimes: prior=%s -> new=%s", prior.run_id, new.run_id)

    touched = storage.record_and_update_flips(
        prior.run_id, new.run_id, alpha=alpha, n_confirm=n_confirm, db_path=db_path
    )

    new_flips = [e for e in touched if e.consecutive_confirmations == 1
                 and e.status == "pending" and e.new_run_id == new.run_id]
    confirmed = [e for e in touched if e.status == "confirmed"]
    reverted = [e for e in touched if e.status == "reverted"]
    held = [e for e in touched if e.status == "pending" and e not in new_flips]

    logger.info(
        "Flip summary: %d newly detected, %d still-pending (held), "
        "%d CONFIRMED, %d reverted (rejected as data-revision/transient).",
        len(new_flips), len(held), len(confirmed), len(reverted),
    )
    for ev in new_flips:
        logger.info("  NEW (provisional)  | %s", _fmt_flip(ev))
    for ev in held:
        logger.info("  HELD (still pending)| %s", _fmt_flip(ev))
    for ev in confirmed:
        logger.info("  CONFIRMED ✓        | %s", _fmt_flip(ev))
    for ev in reverted:
        logger.info("  REVERTED ✗         | %s", _fmt_flip(ev))
    if not touched:
        logger.info("  (no regime status changed between the two runs)")

    logger.info("Monitor cycle done in %.1fs", time.time() - t0)
    return touched


def main() -> None:
    ap = argparse.ArgumentParser(description="Scheduled regime-flip monitor.")
    ap.add_argument("--start-date", default=config.DEFAULT_START_DATE)
    ap.add_argument(
        "--end-date", default=None,
        help="Snapshot end date (YYYY-MM-DD). Omit for the latest available bar.",
    )
    ap.add_argument("--alpha", type=float, default=config.MONITOR_MIN_SIGNIFICANCE_ALPHA)
    ap.add_argument("--n-confirm", type=int, default=config.MONITOR_CONFIRMATION_RUNS)
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument(
        "--loop", action="store_true",
        help="DEMO-ONLY in-process scheduler: repeat every --interval seconds. "
             "Prefer Windows Task Scheduler for production. A fast loop re-runs "
             "the same daily bars, so 'confirmed after N runs' becomes vacuously "
             "true — real confirmation needs real calendar time (see module "
             "docstring).",
    )
    ap.add_argument(
        "--interval", type=int, default=86400,
        help="Seconds between cycles in --loop mode (default 86400 = daily, "
             "matching the data granularity so the confirmation window is "
             "meaningful).",
    )
    args = ap.parse_args()
    _configure_logging()

    def _one() -> None:
        run_monitor_cycle(
            start_date=args.start_date,
            end_date=args.end_date,
            alpha=args.alpha,
            n_confirm=args.n_confirm,
            db_path=Path(args.db),
        )

    if not args.loop:
        _one()
        return

    logger.info("Loop mode: a cycle every %d s. Ctrl-C to stop. "
                "(Windows Task Scheduler is the recommended alternative.)",
                args.interval)
    while True:
        try:
            _one()
        except Exception:  # noqa: BLE001 - a loop must survive one bad cycle
            logger.exception("Monitor cycle failed; will retry next interval.")
        logger.info("Sleeping %d s until the next cycle (%s)...",
                    args.interval,
                    datetime.now(timezone.utc).isoformat())
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
