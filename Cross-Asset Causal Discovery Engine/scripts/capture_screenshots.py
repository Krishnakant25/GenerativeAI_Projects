"""Capture screenshots of the live dashboard for portfolio documentation.

Requires: playwright (already installed), Chromium (playwright install chromium)
Both the API and Streamlit must be running before calling this script.

Usage:
    python scripts/capture_screenshots.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "screenshots"
DASHBOARD_URL = "http://localhost:8501"

# run_20260619_133653_683874bb is the Phase 2 validation run — it has the 106 LLM cards.
# The dropdown label format is "YYYY-MM-DD -> YYYY-MM-DD  (run_id)".
# We match on the run_id suffix to be unambiguous.
TARGET_RUN_SUFFIX = "2026-06-11"   # end date of the Phase 2 run (2018-01-04→2026-06-11)


def wait_for_networkidle(page, timeout_s: int = 12) -> None:
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_s * 1000)
    except Exception:
        pass


def screenshot(page, name: str) -> None:
    path = OUTPUT_DIR / name
    page.screenshot(path=str(path), full_page=True)
    size_kb = path.stat().st_size // 1024
    print(f"  saved {name}  ({size_kb} KB)")


def click_tab(page, label: str) -> bool:
    """Click a top-of-main-content Streamlit tab by its visible text."""
    for selector in [
        f"[data-baseweb='tab-list'] button:has-text('{label}')",
        f"button[role='tab']:has-text('{label}')",
    ]:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=3000):
                el.click()
                time.sleep(3)
                wait_for_networkidle(page)
                return True
        except Exception:
            pass
    return False


def select_run_by_suffix(page, suffix: str) -> bool:
    """Open the run selectbox and choose the option whose label contains `suffix`."""
    try:
        # The selectbox trigger is a <div data-baseweb="select"> inside the sidebar
        # expander. Playwright marks it as "not visible" because it is clipped by
        # the expander's overflow, but it IS interactable — use force=True.
        trigger = page.locator("[data-baseweb='select']").first
        trigger.wait_for(timeout=8000)  # wait for it to exist, not for visibility
        trigger.scroll_into_view_if_needed()
        trigger.click(force=True)
        time.sleep(0.8)

        # Options appear in the popover listbox.
        options = page.locator("[data-baseweb='popover'] li")
        count = options.count()
        print(f"  Dropdown has {count} options")
        for i in range(count):
            opt = options.nth(i)
            text = opt.inner_text()
            print(f"    [{i}] {text.strip()}")
            if suffix in text:
                opt.click()
                time.sleep(0.5)
                print(f"  Selected option containing '{suffix}'")
                return True

        # Nothing matched — pick first
        if count > 0:
            options.first.click()
            print("  No suffix match — picked first option")
        return False
    except Exception as exc:
        print(f"  selectbox error: {exc}")
        return False


def main() -> None:
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, slow_mo=80)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()

        print("Opening dashboard ...")
        page.goto(DASHBOARD_URL, wait_until="domcontentloaded", timeout=30_000)
        wait_for_networkidle(page, 30)
        time.sleep(3)

        screenshot(page, "_debug_initial.png")

        # ---- Select the Phase 2 run (which has the 106 LLM cards) ----------
        print(f"Selecting run with end-date {TARGET_RUN_SUFFIX} ...")
        selected = select_run_by_suffix(page, TARGET_RUN_SUFFIX)

        # ---- Click Load run -------------------------------------------------
        print("Clicking Load run ...")
        try:
            btn = page.get_by_role("button", name="Load run").first
            btn.wait_for(state="visible", timeout=5000)
            btn.click()
            time.sleep(5)
            wait_for_networkidle(page, 15)
        except Exception as exc:
            print(f"  load button error: {exc}")

        screenshot(page, "00_after_load.png")

        # ---- Causal Graph ---------------------------------------------------
        print("Causal Graph ...")
        click_tab(page, "Causal Graph")
        screenshot(page, "01_causal_graph.png")

        # ---- Regime Timeline ------------------------------------------------
        print("Regime Timeline ...")
        click_tab(page, "Regime Timeline")
        screenshot(page, "02_regime_timeline.png")

        # ---- Hypothesis Cards -----------------------------------------------
        print("Hypothesis Cards ...")
        if not click_tab(page, "Hypothesis cards"):
            click_tab(page, "Hypothesis Cards")
        time.sleep(2)
        screenshot(page, "03_hypothesis_cards.png")

        # ---- Business Use Cases --------------------------------------------
        print("Business Use Cases ...")
        if not click_tab(page, "Business use cases"):
            click_tab(page, "Business Use Cases")
        time.sleep(2)
        screenshot(page, "04_business_use_cases.png")

        # ---- Events --------------------------------------------------------
        print("Events ...")
        click_tab(page, "Events")
        screenshot(page, "05_events_tab.png")

        ctx.close()
        browser.close()

    print(f"\nDone. Screenshots saved to: {OUTPUT_DIR}")
    for s in sorted(OUTPUT_DIR.glob("[0-9]*.png")):
        kb = s.stat().st_size // 1024
        print(f"  {s.name}  ({kb} KB)")


if __name__ == "__main__":
    sys.exit(main() or 0)
