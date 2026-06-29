"""Streamlit dashboard — real-time task monitoring and report viewer."""

import json
import logging
import os
from typing import Optional

import httpx
import requests
import streamlit as st

logger = logging.getLogger(__name__)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR", "screenshots")

st.set_page_config(page_title="Agentic Browser", page_icon="🤖", layout="wide")
st.title("🤖 Agentic Browser")

st.session_state.setdefault("task_id", None)
st.session_state.setdefault("stream_log", [])
st.session_state.setdefault("task_status", "idle")
st.session_state.setdefault("final_report", None)
st.session_state.setdefault("is_running", False)
st.session_state.setdefault("last_screenshot_count", 0)

# ── Sidebar: Section 1 — Submit New Task ─────────────────────────────────────
st.sidebar.header("🎯 Submit Task")

goal_input: str = st.sidebar.text_area(
    "Goal",
    key="goal_input",
    height=120,
    placeholder="Research Anthropic's latest funding rounds and key investors",
)
st.sidebar.markdown("**Or use a template:**")
template_choice = st.sidebar.selectbox(
    "Task Template",
    ["None", "News Aggregation", "Multi-Site Research", "Competitive Intelligence"],
    key="template_choice",
)

if template_choice != "None" and st.sidebar.button("Load Template", key="load_template"):
    from agent.task_templates import get_template
    key_map = {
        "News Aggregation": "news_aggregation",
        "Multi-Site Research": "multi_site_research",
        "Competitive Intelligence": "competitive_intelligence",
    }
    template_key = key_map[template_choice]
    topic = st.sidebar.text_input("Topic / Company name:", key="template_topic")
    if topic:
        t = get_template(template_key, topic=topic, company=topic)
        st.session_state.goal_input = t.goal
        st.session_state.max_steps_input = t.max_steps
        st.sidebar.success(f"Template loaded: {t.name}")

max_steps_input: int = st.sidebar.number_input(
    "Max Steps", min_value=5, max_value=50, value=25, step=5, key="max_steps_input"
)
headless_input: bool = st.sidebar.checkbox(
    "Headless mode", value=True, key="headless_input"
)

run_clicked: bool = st.sidebar.button("▶ Run Task", use_container_width=True)

if run_clicked:
    goal = (goal_input or "").strip()
    if not goal:
        st.sidebar.error("Please enter a goal.")
    else:
        st.session_state.stream_log = []
        st.session_state.task_status = "running"
        st.session_state.is_running = True
        with st.spinner("Agent running..."):
            try:
                response = requests.post(
                    f"{API_BASE}/tasks",
                    json={
                        "goal": goal,
                        "max_steps": int(max_steps_input),
                        "headless": headless_input,
                    },
                    stream=True,
                    timeout=300,
                )
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        event_data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    st.session_state.stream_log.append(event_data)
                    event = event_data.get("event")
                    if event == "started":
                        st.session_state.task_id = event_data.get("task_id")
                    elif event == "step":
                        st.sidebar.caption(
                            f"Step {event_data.get('step_number', '?')} — "
                            f"{event_data.get('action', '?')}"
                        )
                    elif event == "finished":
                        st.session_state.task_status = event_data.get("status", "complete")
                        st.session_state.final_report = event_data.get("report")
                        st.session_state.is_running = False
                        break
                    elif event == "error":
                        st.session_state.task_status = "failed"
                        st.session_state.is_running = False
                        st.sidebar.error(event_data.get("error", "Unknown error"))
                        break
            except Exception as exc:  # noqa: BLE001 — surface as sidebar error
                logger.error("run_task streaming failed: %s", exc)
                st.sidebar.error(f"Could not reach API: {exc}")
                st.session_state.task_status = "failed"
                st.session_state.is_running = False
        st.rerun()

# ── Sidebar: Section 2 — Monitor Existing Task (unchanged) ───────────────────
st.sidebar.divider()
st.sidebar.subheader("🔍 Monitor Existing Task")

task_id_input = st.sidebar.text_input(
    "Task ID",
    value=st.session_state.get("task_id") or "",
    help="Paste a task_id from the POST /tasks stream output",
)
if task_id_input.strip():
    st.session_state["task_id"] = task_id_input.strip()

# ── Main area ─────────────────────────────────────────────────────────────────
task_id: Optional[str] = st.session_state.get("task_id") or None

left_col, right_col = st.columns(2)

with left_col:
    st.header("📋 Task Status")
    if task_id is None:
        st.caption("Enter a task ID in the sidebar.")
    elif st.button("Refresh Status"):
        try:
            resp = httpx.get(f"{API_BASE}/tasks/{task_id}", timeout=10)
            if resp.status_code == 200:
                st.json(resp.json())
            else:
                st.error(f"API returned {resp.status_code}: {resp.text}")
        except Exception as exc:  # noqa: BLE001 — show API errors in the UI
            logger.error("status fetch failed: %s", exc)
            st.error(f"Could not reach API: {exc}")

with right_col:
    st.header("📊 Final Report")
    if task_id is None:
        st.caption("No report yet.")
    elif st.button("Load Report"):
        try:
            resp = httpx.get(f"{API_BASE}/tasks/{task_id}/report", timeout=10)
            if resp.status_code == 200:
                report = resp.json()
                findings = report.get("findings") or {}
                summary = (
                    findings.get("summary") if isinstance(findings, dict) else None
                )
                if summary:
                    st.markdown(summary)
                with st.expander("Full report JSON"):
                    st.json(report)
            elif resp.status_code == 404:
                st.warning("No report yet for this task.")
            else:
                st.error(f"API returned {resp.status_code}: {resp.text}")
        except Exception as exc:  # noqa: BLE001 — show API errors in the UI
            logger.error("report fetch failed: %s", exc)
            st.error(f"Could not reach API: {exc}")

    st.header("🔍 Step Trace")
    if task_id is None:
        st.caption("No trace yet.")
    elif st.button("Load Trace"):
        try:
            resp = httpx.get(f"{API_BASE}/tasks/{task_id}/trace", timeout=10)
            if resp.status_code == 200:
                steps = resp.json().get("steps", [])
                if not steps:
                    st.caption("Trace is empty.")
                for s in steps:
                    with st.expander(f"Step {s.get('step')} | {s.get('action')}"):
                        st.json(s)
            elif resp.status_code == 404:
                st.warning("No trace file for this task.")
            else:
                st.error(f"API returned {resp.status_code}: {resp.text}")
        except Exception as exc:  # noqa: BLE001 — show API errors in the UI
            logger.error("trace fetch failed: %s", exc)
            st.error(f"Could not reach API: {exc}")

    st.header("📸 Live Screenshots")

    if st.session_state.task_id is None:
        st.caption("Run a task to see screenshots.")
    else:
        from pathlib import Path
        import glob as glob_mod

        screenshot_dir = Path(SCREENSHOT_DIR) / st.session_state.task_id
        screenshots = sorted(glob_mod.glob(str(screenshot_dir / "step_*.png")))
        # filter out _som and _retry variants for the main viewer
        screenshots = [s for s in screenshots if s.count("_") == 2]

        if not screenshots:
            st.caption("No screenshots yet.")
        else:
            if st.session_state.get("is_running", False):
                latest = screenshots[-1]
                st.image(latest, use_column_width=True,
                         caption=f"Latest: {Path(latest).name}")
                st.caption(f"Total captured: {len(screenshots)} | Auto-refreshing...")
                import time
                time.sleep(3)
                st.rerun()
            else:
                idx = st.slider("Step", 0, len(screenshots) - 1,
                                len(screenshots) - 1, key="screenshot_slider")
                st.image(screenshots[idx], use_column_width=True,
                         caption=Path(screenshots[idx]).name)
                st.caption(f"{len(screenshots)} screenshots captured")

        som_screenshots = sorted(glob_mod.glob(str(screenshot_dir / "*_som.png")))
        if som_screenshots:
            if st.checkbox("Show Set-of-Mark overlays"):
                som_idx = st.slider("SoM Step", 0, len(som_screenshots) - 1, 0,
                                    key="som_slider")
                st.image(som_screenshots[som_idx], use_column_width=True,
                         caption=f"SoM: {Path(som_screenshots[som_idx]).name}")
