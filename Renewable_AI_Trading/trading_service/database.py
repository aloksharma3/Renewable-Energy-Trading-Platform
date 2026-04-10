"""
Database Module (SQLite)
=========================
Stores all trading data in a single file: data/trading.db

TABLES:
    trades     → Every buy/sell/hold decision with full context
    positions  → Current energy inventory (single row, updated on every trade)

WHY SQLITE:
    - Built into Python (import sqlite3), zero installation
    - Single file, no database server needed
    - Survives container restarts if volume is mounted
    - In production, swap to PostgreSQL by changing connection logic

HOW IT'S USED:
    db = Database("data/trading.db")
    db.record_trade("BUY", price=28.5, quantity=50.0, reason="Low price")
    db.get_current_position()  → {"quantity": 50.0, "avg_cost": 28.5, ...}
    db.get_trade_history()     → list of recent trades
    db.get_portfolio_summary() → total profit, trade count, win rate
"""

import sqlite3
import os
from datetime import datetime

class Database:

    def __init__(self, db_path="data/trading.db"):
        """
        Initialize database. Creates file and tables if they don't exist.

        Args:
            db_path: Path to SQLite file. Created automatically.
        """
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self):
        """
        Create a fresh connection for each operation.

        WHY NOT KEEP ONE CONNECTION?
            FastAPI handles multiple requests at once.
            SQLite doesn't handle concurrent writes well with shared connections.
            Fresh connection per request avoids "database is locked" errors.

        row_factory = sqlite3.Row makes results accessible by column name:
            row["price"] instead of row[3]
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # ─── Trades Table ───────────────────────────────
        # Every single trading decision is logged here permanently
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                total_value REAL NOT NULL,
                reason TEXT,
                predicted_price REAL,
                confidence_lower REAL,
                confidence_upper REAL,
                rag_risk_score REAL,
                rag_direction TEXT,
                position_after REAL,
                profit REAL DEFAULT 0
            )
        """)

        # ─── Positions Table ───────────────────────────
        # Single row tracking current portfolio state
        # id = 1 always (CHECK constraint enforces this)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                quantity REAL DEFAULT 0,
                avg_cost REAL DEFAULT 0,
                total_invested REAL DEFAULT 0,
                total_revenue REAL DEFAULT 0,
                realized_profit REAL DEFAULT 0,
                trade_count INTEGER DEFAULT 0,
                sell_count INTEGER DEFAULT 0,
                buy_count INTEGER DEFAULT 0,
                profitable_sells INTEGER DEFAULT 0,
                last_updated TEXT
            )
        """)
        
        # Add this block inside _create_tables() after the positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                price     REAL NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_history_timestamp
            ON price_history (timestamp)
        """)

        # Insert initial position if table is empty
        cursor.execute("SELECT COUNT(*) FROM positions")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO positions (id, quantity, avg_cost, total_invested,
                    total_revenue, realized_profit, trade_count, sell_count,
                    buy_count, profitable_sells, last_updated)
                VALUES (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ?)
            """, (datetime.utcnow().isoformat(),))

        conn.commit()
        conn.close()

    def record_trade(self, action, price, quantity, reason="",
                     predicted_price=None, confidence_lower=None,
                     confidence_upper=None, rag_risk_score=None,
                     rag_direction=None):
        """
        Record a trade and update position.

        POSITION UPDATE LOGIC:
            BUY:
                - Quantity increases
                - Average cost recalculated (weighted average)
                - Example: hold 50 MWh at $30, buy 30 MWh at $28
                  New avg = (50×30 + 30×28) / (50+30) = $29.25

            SELL:
                - Quantity decreases
                - Profit calculated: (sell_price - avg_cost) × quantity
                - Example: avg_cost=$29.25, sell 60 MWh at $72
                  Profit = ($72 - $29.25) × 60 = $2,565

            HOLD:
                - Nothing changes, but logged for audit trail

        Returns:
            dict with trade details and updated position
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        total_value = round(price * quantity, 2)
        profit = 0.0

        # Get current position
        cursor.execute("SELECT * FROM positions WHERE id = 1")
        pos = dict(cursor.fetchone())

        if action == "BUY":
            # Weighted average cost calculation
            old_total = pos["quantity"] * pos["avg_cost"]
            new_total = old_total + total_value
            new_quantity = pos["quantity"] + quantity
            new_avg = new_total / new_quantity if new_quantity > 0 else 0

            cursor.execute("""
                UPDATE positions SET
                    quantity = ?,
                    avg_cost = ?,
                    total_invested = total_invested + ?,
                    trade_count = trade_count + 1,
                    buy_count = buy_count + 1,
                    last_updated = ?
                WHERE id = 1
            """, (round(new_quantity, 2), round(new_avg, 2), total_value, now))

        elif action == "SELL":
            profit = round((price - pos["avg_cost"]) * quantity, 2)
            new_quantity = max(0, pos["quantity"] - quantity)

            # Track profitable sells for win rate calculation
            profitable = 1 if profit > 0 else 0

            cursor.execute("""
                UPDATE positions SET
                    quantity = ?,
                    total_revenue = total_revenue + ?,
                    realized_profit = realized_profit + ?,
                    trade_count = trade_count + 1,
                    sell_count = sell_count + 1,
                    profitable_sells = profitable_sells + ?,
                    last_updated = ?
                WHERE id = 1
            """, (round(new_quantity, 2), total_value, profit, profitable, now))

        elif action == "HOLD":
            cursor.execute("""
                UPDATE positions SET
                    trade_count = trade_count + 1,
                    last_updated = ?
                WHERE id = 1
            """, (now,))

        # Calculate position after trade
        if action == "BUY":
            position_after = pos["quantity"] + quantity
        elif action == "SELL":
            position_after = max(0, pos["quantity"] - quantity)
        else:
            position_after = pos["quantity"]

        # Log the trade
        cursor.execute("""
            INSERT INTO trades (timestamp, action, price, quantity, total_value,
                reason, predicted_price, confidence_lower, confidence_upper,
                rag_risk_score, rag_direction, position_after, profit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, action, price, quantity, total_value, reason,
              predicted_price, confidence_lower, confidence_upper,
              rag_risk_score, rag_direction, round(position_after, 2), profit))

        conn.commit()
        conn.close()

        return {
            "action": action,
            "price": price,
            "quantity": round(quantity, 2),
            "total_value": total_value,
            "profit": profit,
            "reason": reason,
            "position_after": round(position_after, 2),
            "timestamp": now,
        }

    def get_current_position(self):
        """Get current portfolio state."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM positions WHERE id = 1")
        row = dict(cursor.fetchone())
        conn.close()

        # Calculate win rate
        if row["sell_count"] > 0:
            row["win_rate"] = round(row["profitable_sells"] / row["sell_count"] * 100, 1)
        else:
            row["win_rate"] = 0.0

        return row

    def get_trade_history(self, limit=50):
        """Get recent trades, newest first."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM trades ORDER BY id DESC LIMIT ?",
            (limit,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_portfolio_summary(self):
        """
        Get high-level portfolio summary for dashboard.

        Returns:
            {
                "current_position_mwh": 37.0,
                "avg_cost": 29.25,
                "realized_profit": 2847.50,
                "total_trades": 15,
                "buy_count": 8,
                "sell_count": 7,
                "win_rate": 71.4,
                "unrealized_pnl": calculated from current position and market price
            }
        """
        position = self.get_current_position()
        return {
            "current_position_mwh": position["quantity"],
            "avg_cost": position["avg_cost"],
            "total_invested": position["total_invested"],
            "total_revenue": position["total_revenue"],
            "realized_profit": position["realized_profit"],
            "total_trades": position["trade_count"],
            "buy_count": position["buy_count"],
            "sell_count": position["sell_count"],
            "win_rate": position["win_rate"],
            "last_updated": position["last_updated"],
        }

    def reset(self):
        """Reset all data. Used for testing or fresh start on deployment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trades")
        cursor.execute("DELETE FROM positions")
        cursor.execute("""
            INSERT INTO positions (id, quantity, avg_cost, total_invested,
                total_revenue, realized_profit, trade_count, sell_count,
                buy_count, profitable_sells, last_updated)
            VALUES (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ?)
        """, (datetime.utcnow().isoformat(),))
        conn.commit()
        conn.close()
        return {"message": "Database reset to initial state"}
    
    
    def save_price(self, price: float):
        """Persist a market price observation.
        Automatically prunes to keep only the last 96 records (24 hours).
        Called every pipeline cycle alongside update_price_history()."""
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT INTO price_history (timestamp, price) VALUES (?, ?)",
                (datetime.utcnow().isoformat(), round(price, 4))
            )
            # Keep only last 96 records — delete older ones
            conn.execute("""
                DELETE FROM price_history
                WHERE id NOT IN (
                    SELECT id FROM price_history
                    ORDER BY timestamp DESC
                    LIMIT 96
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def get_recent_prices(self, limit: int = 96) -> list:
        """
        Load recent prices for threshold computation on startup.
        Returns list of floats, oldest first — same order as recent_prices[].
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT price FROM price_history ORDER BY timestamp ASC LIMIT ?",
                (limit,)
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def count_prices(self) -> int:
        """Return number of stored price records."""
        conn = self._get_connection()
        try:
            return conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]
        finally:
            conn.close()