import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error
import shap  # ←追加

# =========================
# タイトル
# =========================
st.markdown("<h3>ドル円 AI（RCIレンジ統合・瞬間動き補正版25）</h3>", unsafe_allow_html=True)

# =========================
# タイマー（20秒更新、JSカウントダウンのみ）
# =========================
components.html("""
<div style="font-size:16px; color:white;">
更新: <span id="t">20</span>秒
</div>
<script>
let t = 20;
setInterval(()=>{
  t--;
  if(t<=0)t=20;
  document.getElementById("t").innerText=t;
},1000);
</script>
""", height=35)

st.markdown("<meta http-equiv='refresh' content='20'>", unsafe_allow_html=True)

# =========================
# データ取得と前処理
# =========================
df = yf.download("USDJPY=X", interval="1m", period="1d")
if df is None or len(df) < 50:
    st.error("データ不足")
    st.stop()
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)
df = df.tail(500)

def calc_atr(df, period=14):
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def calc_rci_light(series, period=5, window=60):
    rci = [np.nan]*len(series)
    for i in range(window, len(series)):
        sub = series[i-window:i]
        vol = sub.max() - sub.min()
        adj_period = period
        if vol < 0.03:
            adj_period = max(3, period-2)
        elif vol > 0.1:
            adj_period = period + 2
        w = series[i-adj_period:i]
        rank_price = w.rank().values
        rank_time = np.arange(1, len(w)+1)
        d = rank_price - rank_time
        rci[i] = (1 - (6 * np.sum(d**2)) / (len(w) * (len(w)**2 - 1))) * 100
    return pd.Series(rci, index=series.index)

def create_features(df):
    df["range"] = (df["High"] - df["Low"]).replace(0, np.nan)
    df["body"] = abs(df["Close"] - df["Open"])
    df["upper"] = df["High"] - np.maximum(df["Open"], df["Close"])
    df["lower"] = np.minimum(df["Open"], df["Close"]) - df["Low"]
    df["body_ratio"] = df["body"] / df["range"]
    df["upper_ratio"] = df["upper"] / df["range"]
    df["lower_ratio"] = df["lower"] / df["range"]
    df["buy_pressure"] = df["lower_ratio"] * df["body_ratio"]
    df["sell_pressure"] = df["upper_ratio"] * df["body_ratio"]
    up = (df["Close"] > df["Open"]).astype(int)
    df["streak"] = up.groupby((up == 0).cumsum()).cumsum()
    df["vola"] = df["range"].rolling(5).mean()
    df["vola_change"] = df["vola"].diff()
    df["atr"] = calc_atr(df)
    df["body_atr"] = df["body"] / df["atr"]
    df["round_down"] = np.floor(df["Close"] * 10) / 10
    df["round_up"] = np.ceil(df["Close"] * 10) / 10
    df["dist_down"] = abs(df["Close"] - df["round_down"])
    df["dist_up"] = abs(df["Close"] - df["round_up"])
    df["round_dist"] = np.minimum(df["dist_down"], df["dist_up"]) * 100
    df["round_absorb"] = (df["round_dist"] < 0.03).rolling(5).sum()
    df["round_support"] = ((df["dist_down"] < 0.02) & (df["Close"] > df["Open"])).astype(int)
    df["round_resist"] = ((df["dist_up"] < 0.02) & (df["Close"] < df["Open"])).astype(int)
    df["round_break_up"] = ((df["Close"] > df["round_up"]) & (df["Close"].shift(1) <= df["round_up"])).astype(int)
    df["round_break_down"] = ((df["Close"] < df["round_down"]) & (df["Close"].shift(1) >= df["round_down"])).astype(int)
    df["delta"] = np.where(df["Close"] > df["Close"].shift(1), df["Volume"], -df["Volume"])
    df["cvd"] = df["delta"].cumsum()
    df["cvd_diff"] = df["cvd"].diff()
    df["price_diff"] = df["Close"].diff()
    df["cvd_div"] = df["cvd_diff"] - df["price_diff"]
    df["cvd_absorb_buy"] = ((df["cvd_diff"] > 0) & (df["price_diff"] <= 0)).astype(int)
    df["cvd_absorb_sell"] = ((df["cvd_diff"] < 0) & (df["price_diff"] >= 0)).astype(int)
    df["rci5"] = calc_rci_light(df["Close"], 5)
    df["range_market"] = ((df["vola_change"].abs() < 0.02) & (df["streak"] < 3)).astype(int)
    df["rci_buy"] = ((df["rci5"] < -80) & (df["range_market"] == 1)).astype(int)
    df["rci_sell"] = ((df["rci5"] > 80) & (df["range_market"] == 1)).astype(int)
    df["dir"] = np.where(df["Close"] > df["Close"].shift(1), 1, -1)
    df["body_pips"] = (df["Close"] - df["Open"]) * 100
    df["impact"] = df["Volume"] * df["dir"] * abs(df["body_pips"])
    df["impact_3"] = df["impact"].rolling(3).sum()
    df["impact_acc"] = df["impact_3"].diff()
    df["impact_absorb_buy"] = ((df["impact"] > 0) & (df["Close"] <= df["Close"].shift(1))).astype(int)
    df["impact_absorb_sell"] = ((df["impact"] < 0) & (df["Close"] >= df["Close"].shift(1))).astype(int)
    return df

df = create_features(df)

# =========================
# ターゲット（略そのまま）
shift_steps = 1
move = (df["Close"].shift(-shift_steps) - df["Close"]) * 100
threshold = 0.3
df["y_cls"] = np.where(move > threshold, 1, np.where(move < -threshold, 0, np.nan))
df["y_reg"] = move

shift_steps2 = 2
move2 = (df["Close"].shift(-shift_steps2) - df["Close"]) * 100
df["y_cls2"] = np.where(move2 > threshold, 1, np.where(move2 < -threshold, 0, np.nan))
df["y_reg2"] = move2

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["y_cls","y_reg","y_cls2","y_reg2"]+df.columns.tolist())

# =========================
# 特徴量（そのまま）
cols = [
    "body_ratio","upper_ratio","lower_ratio",
    "buy_pressure","sell_pressure",
    "streak","vola_change","body_atr",
    "round_dist","round_absorb","round_support","round_resist",
    "round_break_up","round_break_down",
    "cvd_div","cvd_absorb_buy","cvd_absorb_sell",
    "impact_3","impact_acc","impact_absorb_buy","impact_absorb_sell"
]
X = df[cols]

split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train_c, y_test_c = df["y_cls"][:split], df["y_cls"][split:]
y_train_r, y_test_r = df["y_reg"][:split], df["y_reg"][split:]
y_train_c2, y_test_c2 = df["y_cls2"][:split], df["y_cls2"][split:]
y_train_r2, y_test_r2 = df["y_reg2"][:split], df["y_reg2"][split:]

model_c = xgb.XGBClassifier(n_estimators=50,max_depth=3,verbosity=0)
model_r = xgb.XGBRegressor(n_estimators=50,max_depth=3,verbosity=0)
model_c2 = xgb.XGBClassifier(n_estimators=50,max_depth=3,verbosity=0)
model_r2 = xgb.XGBRegressor(n_estimators=50,max_depth=3,verbosity=0)

if len(X_train) >= 5:
    model_c.fit(X_train, y_train_c)
    model_r.fit(X_train, y_train_r)
    model_c2.fit(X_train, y_train_c2)
    model_r2.fit(X_train, y_train_r2)

# =========================
# 予測
latest = X.tail(1)
proba = model_c.predict_proba(latest)[0]
pips = model_r.predict(latest)[0]
proba2 = model_c2.predict_proba(latest)[0]
pips2 = model_r2.predict(latest)[0]

# =========================
# SHAP（追加ここだけ）
explainer = shap.TreeExplainer(model_c)
shap_values = explainer.shap_values(latest)
shap_score = float(np.sum(shap_values))

if shap_score > 0.3:
    shap_text = "↑↑↑ 強い上昇"
    shap_color = "red"
elif shap_score > 0.15:
    shap_text = "↑↑ 中上昇"
    shap_color = "red"
elif shap_score > 0.05:
    shap_text = "↑ 弱上昇"
    shap_color = "pink"
elif shap_score < -0.3:
    shap_text = "↓↓↓ 強い下降"
    shap_color = "skyblue"
elif shap_score < -0.15:
    shap_text = "↓↓ 中下降"
    shap_color = "skyblue"
elif shap_score < -0.05:
    shap_text = "↓ 弱下降"
    shap_color = "lightblue"
else:
    shap_text = "・ ノイズ"
    shap_color = "gray"

# =========================
# 瞬間補正（そのまま）
rci_buy = df["rci_buy"].iloc[-1]
rci_sell = df["rci_sell"].iloc[-1]
streak = df["streak"].iloc[-1]

if streak >= 3:
    if rci_buy == 1: proba[0] *= 0.85
    if rci_sell == 1: proba[1] *= 0.85
else:
    if rci_buy == 1: proba[0] *= 0.6
    if rci_sell == 1: proba[1] *= 0.6

total = proba[0]+proba[1]
if total>0:
    proba /= total

if streak >= 3:
    if rci_buy == 1 and pips < 1.0: pips *= 0.8
    if rci_sell == 1 and pips > -1.0: pips *= 0.8
else:
    if rci_buy == 1 and pips < 1.0: pips *= 0.5
    if rci_sell == 1 and pips > -1.0: pips *= 0.5

# =========================
# 色判定（そのまま）
up_color = "red" if proba[1]>=0.7 else "white"
down_color = "skyblue" if proba[0]>=0.7 else "white"
if pips >= 1.5: pips_color = "red"
elif pips <= -1.5: pips_color = "skyblue"
else: pips_color = "white"

up_color2 = "red" if proba2[1]>=0.7 else "white"
down_color2 = "skyblue" if proba2[0]>=0.7 else "white"
if pips2 >= 1.5: pips_color2 = "red"
elif pips2 <= -1.5: pips_color2 = "skyblue"
else: pips_color2 = "white"

# =========================
# 表示
st.subheader("AI判断")

# ←追加（SHAP表示）
st.markdown(
    f"<span style='color:{shap_color}; font-size:20px;'>SHAP強度: {shap_text}</span>",
    unsafe_allow_html=True
)

st.markdown(
f"<span style='color:{up_color};'>上昇確率(60秒): {proba[1]*100:.1f}%</span> / "
f"<span style='color:{up_color2};'>上昇確率(120秒): {proba2[1]*100:.1f}%</span>",
unsafe_allow_html=True)
st.markdown(
f"<span style='color:{down_color};'>下降確率(60秒): {proba[0]*100:.1f}%</span> / "
f"<span style='color:{down_color2};'>下降確率(120秒): {proba2[0]*100:.1f}%</span>",
unsafe_allow_html=True)
st.markdown(
f"<span style='color:{pips_color};'>予想値幅(60秒): {pips:+.2f} pips</span> / "
f"<span style='color:{pips_color2};'>予想値幅(120秒): {pips2:+.2f} pips</span>",
unsafe_allow_html=True)

# 精度表示
if len(X_test) > 0:
    acc = accuracy_score(y_test_c, model_c.predict(X_test))
    mae = mean_absolute_error(y_test_r, model_r.predict(X_test))
    st.write("---")
    acc_color = "#c8a2ff" if acc >= 0.65 else "white"
    st.markdown(f"<span style='color:{acc_color};'>分類精度: {acc:.3f}</span>", unsafe_allow_html=True)
    st.write(f"値幅誤差: {mae:.2f} pips")