"""
Trading Service (v4 — Dynamic Percentile Thresholds)
=====================================================
Makes buy/sell/hold decisions using ML predictions + RAG market intelligence
+ dynamic price-adaptive thresholds.

WHAT CHANGED FROM V3:
    V3: Hardcoded buy/sell thresholds ($30/$55) that didn't adapt to market conditions
    V4: Dynamic thresholds computed from rolling price percentiles
        - Buy threshold = 30th percentile of last 24h prices
        - Sell threshold = 70th percentile of last 24h prices
        - Automatically adapts to summer peaks vs winter lows
        - RAG intelligence further adjusts based on market risk signals

THE DECISION PIPELINE (every 15 minutes):
    1. Get ML prediction from forecast service
       → predicted_price, confidence_lower, confidence_upper
    2. Track actual market price in rolling window (last 96 intervals)
    3. Compute dynamic thresholds from price percentiles
    4. Get RAG assessment → adjust thresholds based on risk/direction
    5. Compare predicted price against dynamic thresholds → BUY/SELL/HOLD
    6. Size the trade based on model confidence
    7. Check position limits
    8. Execute trade and update database
    9. Return decision with full reasoning + threshold context
"""

import os
import logging
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from database import Database

# ─── Setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("trading_service")

app = FastAPI(
    title="Energy Trading Service",
    description="Automated trading with ML predictions, RAG intelligence, and position management",
    version="3.0.0",
)

# ─── Configuration ──────────────────────────────────────────
FORECAST_SERVICE_URL = os.getenv("FORECAST_SERVICE_URL", "http://forecast:8002")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag:8004")
DB_PATH = os.getenv("DB_PATH", "data/trading.db")

# Trading parameters
BASE_TRADE_QUANTITY = float(os.getenv("BASE_TRADE_QUANTITY", "50"))
MAX_POSITION = float(os.getenv("MAX_POSITION_MWH", "500"))
RAG_ADJUSTMENT_FACTOR = float(os.getenv("RAG_ADJUSTMENT_FACTOR", "20"))

# Dynamic threshold config
# Instead of hardcoded buy/sell prices, thresholds are computed from
# recent price history using percentiles. This adapts automatically
# as market conditions change (e.g., summer peak vs winter off-peak).
SELL_PERCENTILE = float(os.getenv("SELL_PERCENTILE", "70"))   # sell above 70th percentile
BUY_PERCENTILE = float(os.getenv("BUY_PERCENTILE", "30"))     # buy below 30th percentile
FALLBACK_SELL = float(os.getenv("SELL_THRESHOLD", "40"))       # fallback if no history
FALLBACK_BUY = float(os.getenv("BUY_THRESHOLD", "25"))        # fallback if no history

# ─── Initialize Database ────────────────────────────────────
db = Database(DB_PATH)

# ─── Price History for Dynamic Thresholds ───────────────────
recent_prices = []  # rolling window of recent market prices


# ═══════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════

def update_price_history(price):
    """
    Track recent market prices for dynamic threshold computation.
    Keeps a rolling window of the last 96 prices (~24 hours at 15-min intervals).
    """
    if price is not None and price > 0:
        recent_prices.append(float(price))
        while len(recent_prices) > 96:
            recent_prices.pop(0)


def compute_dynamic_thresholds():
    """
    Compute buy/sell thresholds from recent price percentiles.

    WHY DYNAMIC THRESHOLDS?
        Hardcoded thresholds (e.g., sell at $55, buy at $30) fail when
        market conditions change. In winter, prices might range $15-35.
        In summer, $40-120. A fixed $55 sell threshold would never
        trigger in winter and trigger too often in summer.

    HOW IT WORKS:
        - Collect last 24 hours of market prices (96 data points)
        - Buy threshold = 30th percentile (bottom of recent range)
        - Sell threshold = 70th percentile (top of recent range)
        - Thresholds automatically adjust as prices shift

    EXAMPLE:
        Recent prices: [$18, $20, $22, $25, $28, $30, $35, $40, $45, $50]
        30th percentile = $22.60 → buy below this (cheap relative to recent)
        70th percentile = $41.50 → sell above this (expensive relative to recent)

    Falls back to configured defaults if not enough price history yet.
    """
    if len(recent_prices) < 10:
        return FALLBACK_SELL, FALLBACK_BUY

    import numpy as np
    prices = np.array(recent_prices)
    sell_threshold = round(float(np.percentile(prices, SELL_PERCENTILE)), 2)
    buy_threshold = round(float(np.percentile(prices, BUY_PERCENTILE)), 2)

    # Ensure minimum gap between buy and sell ($3 minimum spread)
    if sell_threshold - buy_threshold < 3:
        mid = (sell_threshold + buy_threshold) / 2
        sell_threshold = round(mid + 1.5, 2)
        buy_threshold = round(mid - 1.5, 2)

    return sell_threshold, buy_threshold

def get_forecast():
    """
    Get latest ML prediction from forecast service.

    Returns:
        {
            "price": {"predicted": 52.0, "confidence_lower": 45.0, "confidence_upper": 59.0},
            "energy_output": {"predicted": 85.0, ...},
            "demand": {"predicted": 950.0, ...},
            "actual_market_price": 50.5,
            "dam_price": 48.50
        }
    """
    try:
        response = requests.get(f"{FORECAST_SERVICE_URL}/forecast/latest", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Forecast service unavailable: {e}")
        return None


def get_rag_assessment():
    """
    Get market risk assessment from RAG service.

    Returns:
        {
            "risk_score": 0.85,
            "risk_level": "high",
            "price_direction": "up",
            "factors": ["conservation alert", "generator outage"],
            "summary": "Multiple indicators suggest..."
        }
    """
    try:
        response = requests.post(
            f"{RAG_SERVICE_URL}/analyze",
            json={"query": "What factors might affect ERCOT HB_NORTH electricity prices in the next few hours?"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"RAG service unavailable: {e}")
        # Return neutral assessment — trading continues without RAG
        return {
            "risk_score": 0.5,
            "risk_level": "medium",
            "price_direction": "stable",
            "factors": ["RAG unavailable"],
            "summary": "RAG service not available, using default assessment",
            "fallback": True,
        }


def adjust_thresholds(rag_assessment):
    """
    Compute trading thresholds using dynamic percentiles + RAG adjustment.

    TWO-LAYER THRESHOLD SYSTEM:

    Layer 1 — Dynamic Base (from market data):
        sell_threshold = 70th percentile of last 24h prices
        buy_threshold  = 30th percentile of last 24h prices
        These adapt automatically as market conditions change.

    Layer 2 — RAG Adjustment (from market intelligence):
        If RAG detects high risk + upward direction:
            → lower sell threshold (sell earlier before spike)
            → raise buy threshold (don't buy into rising market)
        If RAG detects high risk + downward direction:
            → raise sell threshold (wait for recovery)
            → lower buy threshold (buy the dip)

    Returns:
        (sell_threshold, buy_threshold)
    """
    # Layer 1: Dynamic percentile-based thresholds
    sell_threshold, buy_threshold = compute_dynamic_thresholds()

    # Layer 2: RAG adjustment
    risk_score = rag_assessment.get("risk_score", 0.5)
    direction = rag_assessment.get("price_direction", "stable")

    if risk_score > 0.6:
        adjustment = risk_score * RAG_ADJUSTMENT_FACTOR

        if direction == "up":
            sell_threshold -= adjustment
            buy_threshold += adjustment
        elif direction == "down":
            sell_threshold += adjustment
            buy_threshold -= adjustment

    return round(sell_threshold, 2), round(buy_threshold, 2)


def calculate_trade_quantity(confidence_lower, confidence_upper, action):
    """
    Size the trade based on model confidence.

    WHY THIS MATTERS:
        If ML predicts price=$52 with confidence [$50, $54] (tight, $4 range)
        → Model is very sure → trade more

        If ML predicts price=$52 with confidence [$35, $69] (wide, $34 range)
        → Model is uncertain → trade less

    FORMULA:
        confidence_width = upper - lower
        width_ratio = width / predicted_price
        quantity = base_quantity × (1 - width_ratio)

        Clamped between 10 MWh (minimum) and base_quantity (maximum)

    Args:
        confidence_lower: Lower bound of prediction
        confidence_upper: Upper bound of prediction
        action: "BUY" or "SELL"

    Returns:
        Trade quantity in MWh
    """
    if confidence_lower is None or confidence_upper is None:
        return BASE_TRADE_QUANTITY

    width = confidence_upper - confidence_lower
    midpoint = (confidence_upper + confidence_lower) / 2

    if midpoint <= 0:
        return BASE_TRADE_QUANTITY

    # width_ratio: 0 = perfectly confident, 1 = extremely uncertain
    width_ratio = min(width / midpoint, 1.0)

    # Scale quantity: confident → full size, uncertain → reduced
    quantity = BASE_TRADE_QUANTITY * (1 - width_ratio * 0.7)

    # Minimum 10 MWh, maximum BASE_TRADE_QUANTITY
    quantity = max(10, min(quantity, BASE_TRADE_QUANTITY))

    return round(quantity, 2)


def check_position_limits(action, quantity):
    """
    Ensure trade doesn't violate position limits.

    Rules:
        - Can't sell more than you hold
        - Can't buy if total position would exceed MAX_POSITION
        - Returns adjusted quantity (may be reduced)

    Args:
        action: "BUY" or "SELL"
        quantity: Desired trade quantity

    Returns:
        Adjusted quantity (may be 0 if trade is not possible)
    """
    position = db.get_current_position()
    current_qty = position["quantity"]

    if action == "SELL":
        if current_qty <= 0:
            return 0  # Nothing to sell
        return min(quantity, current_qty)  # Can't sell more than held

    elif action == "BUY":
        room = MAX_POSITION - current_qty
        if room <= 0:
            return 0  # Position full
        return min(quantity, room)  # Can't exceed max

    return quantity


# ═══════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health():
    position = db.get_current_position()
    return {
        "status": "healthy",
        "service": "trading",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "current_position_mwh": position["quantity"],
        "realized_profit": position["realized_profit"],
        "total_trades": position["trade_count"],
    }


@app.post("/trade/execute")
def execute_trade():
    """
    Run the full trading pipeline.

    Called by scheduler every 15 minutes:
        curl -X POST http://trading:8003/trade/execute

    Pipeline:
        1. Get ML forecast (predicted price + confidence interval)
        2. Get RAG assessment (risk score + direction)
        3. Adjust thresholds based on RAG
        4. Calculate trade size based on confidence
        5. Check position limits
        6. Execute and log to database
        7. Return decision with full reasoning

    Returns complete decision context so dashboard can show
    exactly WHY each trade was made.
    """
    # Step 1: Get ML forecast
    forecast = get_forecast()
    if not forecast:
        return {"action": "HOLD", "reason": "Forecast service unavailable"}

    price_data = forecast.get("price", {})
    predicted_price = price_data.get("predicted", 0)
    confidence_lower = price_data.get("confidence_lower")
    confidence_upper = price_data.get("confidence_upper")
    actual_price = forecast.get("actual_market_price")

    # Track actual market price for dynamic threshold computation
    update_price_history(actual_price)

    if predicted_price <= 0:
        return {"action": "HOLD", "reason": "Invalid price prediction"}

    # Step 2: Get RAG assessment
    rag = get_rag_assessment()

    # Step 3: Compute dynamic thresholds (percentile-based + RAG adjustment)
    sell_threshold, buy_threshold = adjust_thresholds(rag)

    # Step 4: Determine action
    if predicted_price >= sell_threshold:
        action = "SELL"
        reason = f"Price ${predicted_price} >= sell threshold ${sell_threshold}"
    elif predicted_price <= buy_threshold:
        action = "BUY"
        reason = f"Price ${predicted_price} <= buy threshold ${buy_threshold}"
    else:
        action = "HOLD"
        reason = f"Price ${predicted_price} between buy ${buy_threshold} and sell ${sell_threshold}"

    # Add RAG context to reason
    if rag.get("risk_score", 0.5) > 0.6:
        reason += f" | RAG: {rag['risk_level']} risk ({rag['risk_score']:.2f}), direction={rag['price_direction']}"

    # Step 5: Calculate quantity and check limits
    if action != "HOLD":
        quantity = calculate_trade_quantity(confidence_lower, confidence_upper, action)
        quantity = check_position_limits(action, quantity)

        if quantity <= 0:
            position = db.get_current_position()
            if action == "SELL":
                reason = "Nothing to sell (position is 0)"
            else:
                reason = f"Position full ({position['quantity']}/{MAX_POSITION} MWh)"
            action = "HOLD"
            quantity = 0
    else:
        quantity = 0

    # Step 6: Record trade
    trade_price = actual_price if actual_price else predicted_price
    result = db.record_trade(
        action=action,
        price=trade_price,
        quantity=quantity,
        reason=reason,
        predicted_price=predicted_price,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        rag_risk_score=rag.get("risk_score"),
        rag_direction=rag.get("price_direction"),
    )

    # Step 7: Build response
    base_sell, base_buy = compute_dynamic_thresholds()
    response = {
        **result,
        "thresholds": {
            "sell": sell_threshold,
            "buy": buy_threshold,
            "base_sell": base_sell,
            "base_buy": base_buy,
            "method": "dynamic_percentile",
            "price_history_depth": len(recent_prices),
            "sell_percentile": SELL_PERCENTILE,
            "buy_percentile": BUY_PERCENTILE,
        },
        "forecast": {
            "predicted_price": predicted_price,
            "confidence_lower": confidence_lower,
            "confidence_upper": confidence_upper,
            "actual_market_price": actual_price,
        },
        "rag": {
            "risk_score": rag.get("risk_score"),
            "risk_level": rag.get("risk_level"),
            "direction": rag.get("price_direction"),
            "factors": rag.get("factors", []),
        },
        "dam": {
            "dam_price": forecast.get("dam_price"),
            "rt_dam_spread": forecast.get("rt_dam_spread"),
        },
    }

    logger.info(
        f"Trade: {action} {quantity}MWh at ${trade_price} | "
        f"Thresholds: buy=${buy_threshold} sell=${sell_threshold} (dynamic, {len(recent_prices)} prices) | "
        f"Reason: {reason} | Profit: ${result['profit']}"
    )
    return response


@app.get("/position")
def get_position():
    """Get current portfolio position."""
    return db.get_current_position()


@app.get("/trades")
def get_trades(limit: int = 50):
    """Get recent trade history."""
    return db.get_trade_history(limit=limit)


@app.get("/portfolio")
def get_portfolio():
    """Get portfolio summary for dashboard."""
    return db.get_portfolio_summary()


@app.post("/reset")
def reset_portfolio():
    """Reset all trades and position. Used for fresh start."""
    return db.reset()