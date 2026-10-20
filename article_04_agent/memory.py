import sqlite3
from datetime import datetime


class MemoryManager:
    def __init__(self, db_path: str = "research.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._initialize_db()

    def _initialize_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS research_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                topic       TEXT NOT NULL,
                query       TEXT NOT NULL,
                summary     TEXT,
                details     TEXT,
                follow_ups  TEXT,
                confidence_score  INTEGER,
                source_reference  TEXT,
                timestamp   TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def save_episode(
        self,
        topic: str,
        query: str,
        summary: str,
        details: str,
        follow_ups: str = None,
        confidence_score: int = None,
        source_reference: str = None
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO research_sessions
                (topic, query, summary, details, follow_ups, confidence_score, source_reference, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (topic, query, summary, details, follow_ups, confidence_score, source_reference,
             datetime.now().isoformat())
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_all_topics(self) -> list[dict]:
        cursor = self.conn.execute(
            """
            SELECT id, topic, summary, timestamp
            FROM research_sessions
            ORDER BY timestamp DESC
            LIMIT 100
            """
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_episodes_by_ids(self, ids: list[int]) -> list[dict]:
        placeholders = ",".join("?" * len(ids))
        cursor = self.conn.execute(
            f"SELECT * FROM research_sessions WHERE id IN ({placeholders})",
            ids
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        self.conn.close()