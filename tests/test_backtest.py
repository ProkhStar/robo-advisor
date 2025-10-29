import pandas as pd
import numpy as np
from backend.app.services.backtest import backtest_weights

def test_backtest_simple():
    # pequeno dataset sintético: 5 dias, 2 ativos
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.DataFrame({
        "AAA": [100, 101, 102, 101, 103],
        "BBB": [50, 49, 51, 52, 50],
    }, index=dates)
    weights = {"AAA": 0.6, "BBB": 0.4}
    pv, metrics = backtest_weights(prices, weights, capital=1000, rebalance="monthly", fee=0.0)
    assert pv.shape[0] == 5
    assert "cagr" in metrics
