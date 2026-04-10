"""
Forecast Database (SQLite)
===========================
Persists every forecast record so that history survives container restarts.

Without this, forecast_history[] is a plain Python list that resets to [] on
every `docker compose up`, meaning the backtest engine and trend charts start
empty every time.

TABLES
    forecast_history  → one row per pipeline run (every 15 min)

FILE LOCATION
    Stored at DB_PATH (default: data/forecast.db)
    Mounted via the forecast_data Docker volume defined in docker-compose.yml
    → data survives restarts and docker compose down / up cycles

USAGE
    db = ForecastDatabase("data/forecast.db")

    # Save a forecast produced by the pipeline
    db.save_forecast(result_dict)

    # Retrieve history for the dashboard / backtest engine
    records = db.get_history(limit=1500)   # newest first

    # Retrieve history oldest-first (for backtest)
    records = db.get_history(limit=1500, order="asc")

SCHEMA NOTES
    - payload column stores the full forecast JSON (energy, price, demand,
      weather, confidence intervals, etc.) so we never lose any fields
    - key scalar fields (timestamp, predicted_price, actual_market_price,
      dam_price) are also stored as dedicated columns for fast queries and
      easy SQL inspection without parsing JSON
    - actual_market_price is the ERCOT RT price at the time of the forecast
      and is what the backtest engine uses as the "true" price
"""

import sqlite3
import json
import os
from datetime import datetime


class ForecastDatabase:

    def __init__(self, db_path: str = "data/forecast.db"):
        """
        Initialise the database.  Creates the file and table if they don't
        exist yet.  Safe to call on every service startup.

        Args:
            db_path: Path to the SQLite file.  Parent directory is created
                     automatically if missing.
        """
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self.db_path = db_path
        self._create_tables()

    # ─── Connection ─────────────────────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        """
        Return a fresh connection for each operation.

        A fresh connection per call (rather than one shared connection) avoids
        "database is locked" errors when FastAPI handles concurrent requests.
        row_factory = sqlite3.Row lets callers access columns by name.
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode — allows concurrent reads while a write is in progress
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ─── Schema ─────────────────────────────────────────────────────────────

    def _create_tables(self):
        """Create tables if they don't exist.  Idempotent — safe on restart."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecast_history (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,

                    -- When the forecast was produced (ISO-8601 string, UTC)
                    timestamp            TEXT NOT NULL,

                    -- Key scalar fields extracted for fast access
                    predicted_price      REAL,
                    confidence_lower     REAL,
                    confidence_upper     REAL,
                    actual_market_price  REAL,
                    dam_price            REAL,
                    predicted_energy     REAL,
                    predicted_demand     REAL,

                    -- Full forecast payload (JSON) — preserves all fields
                    payload              TEXT NOT NULL,

                    -- Row insertion time (differs from timestamp if pipeline is slow)
                    created_at           TEXT NOT NULL
                )
            """)

            # Index on timestamp for fast range queries used by the dashboard
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_forecast_timestamp
                ON forecast_history (timestamp)
            """)

            conn.commit()
        finally:
            conn.close()

    # ─── Write ──────────────────────────────────────────────────────────────

    def save_forecast(self, forecast: dict) -> int:
        """
        Persist a forecast record produced by the pipeline.

        Args:
            forecast: The dict returned by POST /forecast/run.
                      Must contain at least a "timestamp" key.

        Returns:
            The rowid of the newly inserted row.
        """
        timestamp = forecast.get("timestamp", datetime.utcnow().isoformat())

        # Extract scalar fields (None if missing — columns are nullable)
        price_data   = forecast.get("price", {}) or {}
        energy_data  = forecast.get("energy_output", {}) or {}
        demand_data  = forecast.get("demand", {}) or {}

        predicted_price  = price_data.get("predicted")
        confidence_lower = price_data.get("confidence_lower")
        confidence_upper = price_data.get("confidence_upper")
        actual_price     = forecast.get("actual_market_price")
        dam_price        = forecast.get("dam_price")
        predicted_energy = energy_data.get("predicted")
        predicted_demand = demand_data.get("predicted")

        payload    = json.dumps(forecast)
        created_at = datetime.utcnow().isoformat()

        conn = self._get_connection()
        try:
            cursor = conn.execute("""
                INSERT INTO forecast_history (
                    timestamp, predicted_price, confidence_lower, confidence_upper,
                    actual_market_price, dam_price, predicted_energy, predicted_demand,
                    payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, predicted_price, confidence_lower, confidence_upper,
                actual_price, dam_price, predicted_energy, predicted_demand,
                payload, created_at,
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    # ─── Read ────────────────────────────────────────────────────────────────

    def get_history(self, limit: int = 96, order: str = "desc") -> list[dict]:
        """
        Return recent forecast records as a list of dicts.

        The payload column is parsed back to a dict so callers receive the
        same structure as the original forecast dict — no schema changes
        needed in the dashboard or backtest engine.

        Args:
            limit: Maximum number of records to return.
            order: "desc" → newest first (default, for dashboard charts)
                   "asc"  → oldest first (for backtest engine)

        Returns:
            List of forecast dicts, ordered as requested.
        """
        direction = "DESC" if order.lower() != "asc" else "ASC"

        conn = self._get_connection()
        try:
            cursor = conn.execute(
                f"SELECT payload FROM forecast_history "
                f"ORDER BY timestamp {direction} LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        records = []
        for row in rows:
            try:
                records.append(json.loads(row["payload"]))
            except (json.JSONDecodeError, TypeError):
                continue  # skip any corrupted rows silently

        return records

    def get_latest(self) -> dict | None:
        """
        Return the most recent forecast record, or None if the table is empty.
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT payload FROM forecast_history ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        try:
            return json.loads(row["payload"])
        except (json.JSONDecodeError, TypeError):
            return None

    def count(self) -> int:
        """Return total number of stored forecast records."""
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM forecast_history")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    # ─── Maintenance ────────────────────────────────────────────────────────

    def prune(self, keep: int = 10_000):
        """
        Delete the oldest records, keeping only the most recent `keep` rows.

        At 96 records/day, 10,000 rows ≈ 104 days of history (~3.5 months).
        Call this periodically (e.g. weekly via the scheduler) to prevent
        unbounded database growth.

        Args:
            keep: Number of most-recent rows to retain.
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                DELETE FROM forecast_history
                WHERE id NOT IN (
                    SELECT id FROM forecast_history
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
            """, (keep,))
            conn.commit()
        finally:
            conn.close()

    def reset(self):
        """
        Delete all rows.  Useful for testing or a clean-slate restart.
        Does NOT drop the table — the schema is preserved.
        """
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM forecast_history")
            conn.commit()
        finally:
            conn.close()
        return {"message": "Forecast history cleared"}