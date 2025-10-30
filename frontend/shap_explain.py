import streamlit as st
import tempfile
import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# caching heavy ops
@st.cache_data(show_spinner=False)
def fetch_prices_cached(tickers, start):
    return yf.download(tickers, start=start, auto_adjust=True)["Close"]

def build_features(prices, forward_days=21):
    features = []
    for t in prices.columns:
        s = prices[t].dropna()
        df = pd.DataFrame(index=s.index)
        df['ret_5'] = s.pct_change(5)
        df['ret_21'] = s.pct_change(21)
        df['ret_63'] = s.pct_change(63)
        df['vol_21'] = s.pct_change().rolling(21).std()
        df['vol_63'] = s.pct_change().rolling(63).std()
        df['ma_21'] = s.rolling(21).mean() / s - 1
        df['ma_63'] = s.rolling(63).mean() / s - 1
        df['zscore_21'] = (s - s.rolling(21).mean()) / s.rolling(21).std()
        df['forward_ret'] = s.pct_change(forward_days).shift(-forward_days)
        df['asset'] = t
        df = df.dropna()
        if df.shape[0] > 0:
            features.append(df)
    if len(features) == 0:
        return None, None
    allf = pd.concat(features, axis=0)
    assets_dummies = pd.get_dummies(allf['asset'], prefix='asset')
    X = pd.concat([allf.drop(columns=['forward_ret','asset']), assets_dummies], axis=1)
    y = allf['forward_ret']
    return X, y

def render_shap(tickers, start_date, forward_days=21, sample_frac=0.2):
    st.info("A preparar dados (isto pode demorar alguns segundos)...")
    prices = fetch_prices_cached(tickers, start_date)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.dropna(axis=1, how="all")
    if prices.shape[1] == 0:
        st.error("Nenhum dado para os tickers indicados.")
        return

    X, y = build_features(prices, forward_days=forward_days)
    if X is None or y is None:
        st.error("Não foi possível construir features com os dados fornecidos.")
        return

    # sample to speed up
    if sample_frac < 1.0:
        Xs, _, ys, _ = train_test_split(X, y, test_size=1-sample_frac, random_state=42)
    else:
        Xs, ys = X, y

    st.write("Dados prontos:", Xs.shape)

    # train simple model
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(Xs, ys)

    st.write("Modelo treinado. A calcular SHAP values (pode demorar)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    # summary plot -> salvar imagem temporária
    fig = plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, Xs, show=False)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)

    st.image(tmp.name, caption="SHAP summary (features)")

    # importance table (mean abs shap)
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": mean_abs})
    feat_imp = feat_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    st.subheader("Feature importance (mean |SHAP|)")
    st.dataframe(feat_imp.head(30))

    # cleanup file
    try:
        os.remove(tmp.name)
    except Exception:
        pass
