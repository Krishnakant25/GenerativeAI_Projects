"""SQLite persistence for Screen Copilot observations and suggestions."""

import aiosqlite
from screen_copilot.models import Observation, Suggestion

DB_PATH = "screen_copilot/copilot.db"


async def init_db() -> None:
    """Create observations and suggestions tables if they do not exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                window_title TEXT,
                app_name TEXT,
                screenshot_path TEXT,
                vision_summary TEXT,
                suggested_action TEXT,
                confidence TEXT,
                session_id TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS suggestions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                suggestion_text TEXT,
                category TEXT,
                confidence TEXT,
                based_on_observations INTEGER,
                session_id TEXT
            )
        """)
        await db.commit()


async def save_observation(obs: Observation, session_id: str) -> None:
    """Insert an Observation row into the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO observations
                (id, timestamp, window_title, app_name, screenshot_path,
                 vision_summary, suggested_action, confidence, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                obs.id,
                obs.timestamp.isoformat(),
                obs.window_title,
                obs.app_name,
                obs.screenshot_path,
                obs.vision_summary,
                obs.suggested_action,
                obs.confidence,
                session_id,
            ),
        )
        await db.commit()


async def save_suggestion(sug: Suggestion, session_id: str) -> None:
    """Insert a Suggestion row into the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO suggestions
                (id, timestamp, suggestion_text, category,
                 confidence, based_on_observations, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sug.id,
                sug.timestamp.isoformat(),
                sug.suggestion_text,
                sug.category,
                sug.confidence,
                sug.based_on_observations,
                session_id,
            ),
        )
        await db.commit()


async def get_recent_suggestions(limit: int = 10) -> list[dict]:
    """Return the most recent suggestions, newest first."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM suggestions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]
