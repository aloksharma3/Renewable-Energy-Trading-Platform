from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TradeRequest(BaseModel):
    predicted_output: float  # in MW
    predicted_price: float   # in USD/MWh
    interval_minutes: int = 5  # default interval

@app.post("/trade")
def trade(request: TradeRequest):
    output = request.predicted_output
    price = request.predicted_price
    interval = request.interval_minutes

    # Calculate quantity in MWh
    quantity = output * (interval / 60.0)

    if price < 35:
        decision = "Buy"
        reason = "Low market price"
        profit = -1 * price * quantity  # Buying = spending
    elif price > 80:
        decision = "Sell"
        reason = "High market price"
        profit = price * quantity
    else:
        decision = "Hold"
        reason = "Moderate market conditions"
        profit = 0.0
        quantity = 0.0

    return {
        "decision": decision,
        "reason": reason,
        "profit": round(profit, 2),
        "quantity_mwh": round(quantity, 2)
    }
