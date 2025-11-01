def trading_decision(predicted_output, market_price, threshold=40):
    """
    Returns a tuple: (decision, reason)
    """
    if predicted_output > 100 and market_price > threshold:
        return "Sell", "High output & good price"
    elif market_price < threshold - 5:
        return "Buy", "Low market price"
    else:
        return "Hold", "Not profitable (low output or price)"
