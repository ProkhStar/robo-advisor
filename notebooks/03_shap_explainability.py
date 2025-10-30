# %% [markdown]
# 03 - SHAP Explainability (skeleton)
#
# Este notebook calcula features por ativo, treina um modelo simples (RandomForest)
# para prever retornos futuros e calcula valores SHAP para explicar as previsões.
#
# Uso: abrir no VSCode/Jupyter (Jupytext .py -> notebook) e correr as células.

# %%
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# %%
def fetch_prices(tickers, start):
    data = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(axis=1, how="all")
    return data

# %%
def build_features(prices, forward_days=21):
    # prices: DataFrame indexed by date, columns = tickers
    features = []
    targets = []
    idx = []
    for t in prices.columns:
        s = prices[t].dropna()
        df = pd.DataFrame(index=s.index)
        # short / mid / long momentum
        df['ret_5'] = s.pct_change(5)
        df['ret_21'] = s.pct_change(21)
        df['ret_63'] = s.pct_change(63)
        # vol windows
        df['vol_21'] = s.pct_change().rolling(21).std()
        df['vol_63'] = s.pct_change().rolling(63).std()
        # moving averages
        df['ma_21'] = s.rolling(21).mean() / s - 1
        df['ma_63'] = s.rolling(63).mean() / s - 1
        # zscore momentum
        df['zscore_21'] = (s - s.rolling(21).mean()) / s.rolling(21).std()
        # forward return target
        df['forward_ret'] = s.pct_change(forward_days).shift(-forward_days)
        # asset id (one-hot later)
        df['asset'] = t
        df = df.dropna()
        if df.shape[0] == 0:
            continue
        features.append(df)
    if len(features) == 0:
        return None, None
    allf = pd.concat(features, axis=0)
    # one-hot asset
    assets_dummies = pd.get_dummies(allf['asset'], prefix='asset')
    X = pd.concat([allf.drop(columns=['forward_ret','asset']), assets_dummies], axis=1)
    y = allf['forward_ret']
    return X, y

# %%
# Exemplo de uso:
# tickers = ["SPY","AGG","VT","EEM","BND"]
# prices = fetch_prices(tickers, "2015-01-01")
# X, y = build_features(prices, forward_days=21)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
# model.fit(X_train, y_train)
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, show=True)
