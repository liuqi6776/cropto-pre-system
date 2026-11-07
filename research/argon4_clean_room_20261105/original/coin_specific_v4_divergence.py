#!/usr/bin/env python3
"""
Coin-Specific Causal Structural Divergence (V2/V3) + V4 Magnitude-Gap Divergence Builder
(`coin_specific_v4_divergence.py`)
========================================================================================
For any target asset (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`), computes:
  1. Target Coin's OWN 5m / 1h / 4h Causal Fibonacci Retracement (23.6% rebound trigger,
     0% extreme stop break, 1h 38.2% take-profit room/reward-to-risk).
  2. Target Coin's OWN Causal Structural Divergences (DIF, RSI-14, Volume regular divergence
     at 1h S/R vs pure, hidden divergence without 23.6% gate, MACD near-cross rejection
     "将穿未穿" and slight-cross snap-back "轻微穿过又收回").
  3. Target Coin's OWN V4 Magnitude-Gap Divergences ("价格变化幅度与指标变化幅度差距过大"):
     - Rule R1: Proportional DIF/RSI-vs-Price Magnitude Gap + 23.6% Rebound Hook
     - Rule R2: Order-Flow Effort vs. Price Result Absorption Gap (w = 3, 6, 12, 24)
     - Rule R3: Trade-Velocity vs. Price Magnitude Gap (w = 3, 6, 12)
     - Rule R4: DIF/RSI Channel Position vs. Price Channel Position Discrepancy (w = 18, 36, 72)
  4. Cross-Asset Target-vs-BTC Relative Magnitude-Gap Divergence (capturing when BTC has
     already moved / diverged while ETH/SOL/BNB has not yet caught up).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff(1)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return (100.0 - (100.0 / (1.0 + avg_gain / (avg_loss + 1e-12)))).fillna(50.0)


def _roll_z(s: pd.Series, w_norm: int = 288) -> pd.Series:
    mu = s.rolling(w_norm, min_periods=24).mean()
    sd = s.rolling(w_norm, min_periods=24).std() + 1e-8
    return ((s - mu) / sd).clip(-5.0, 5.0)


def _soft_gate(g: pd.Series, tau: float = 1.15) -> pd.Series:
    """Non-linear tail gate: suppresses small tracking differences |g| < tau, amplifies large gaps |g| > tau."""
    return g * np.tanh((g / tau) ** 2)


def build_coin_own_structural_and_v4_mag_gap(
    df_coin: pd.DataFrame,
    df_btc: pd.DataFrame,
    ofi_4c: pd.Series,
) -> pd.DataFrame:
    """Build complete coin-specific V2/V3 structural + V4 Magnitude-Gap features on `df_coin.index`."""
    idx = df_coin.index
    c = df_coin["close"].astype(float)
    o = df_coin["open"].astype(float)
    h = df_coin["high"].astype(float)
    l = df_coin["low"].astype(float)
    v = df_coin["volume"].astype(float)
    qv = df_coin["quote_volume"].astype(float)
    tbqv = df_coin["taker_buy_quote_volume"].astype(float)
    cnt = df_coin["count"].astype(float)

    prev_c = c.shift(1).fillna(c)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    atr14 = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=5).mean() + 1e-8

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    dif = (ema12 - ema26) / atr14
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    rsi14 = (_wilder_rsi(c, 14) - 50.0) / 15.0
    vol_z = _roll_z(np.log(v + 1.0), 144)

    ofi_coin = (2.0 * tbqv - qv) / (qv + 1e-12)
    ofi_4c_al = ofi_4c.reindex(idx).fillna(0.0)
    clv_coin = (2.0 * c - h - l) / (h - l + 1e-8)

    cols = {}

    # =========================================================================
    # BLOCK A: Coin's OWN 5m / 1h / 4h Causal Fibonacci Structure (23.6% Rebound + 38.2% TP Room)
    # =========================================================================
    for tf_name, win in [("5m", 36), ("1h", 144), ("4h", 576)]:
        r_hi = h.rolling(win, min_periods=12).max()
        r_lo = l.rolling(win, min_periods=12).min()
        span = (r_hi - r_lo).clip(lower=1e-8)
        f_pos = (c - r_lo) / span
        cols[f"own_fib_pos_{tf_name}"] = f_pos

        touch_lo = (f_pos.rolling(6, min_periods=1).min() <= 0.14).astype(float)
        trig_bull_236 = touch_lo * ((f_pos >= 0.236) & (f_pos <= 0.618) & (hist > hist.shift(1))).astype(float)

        touch_hi = (f_pos.rolling(6, min_periods=1).max() >= 0.86).astype(float)
        trig_bear_236 = touch_hi * ((f_pos <= 0.764) & (f_pos >= 0.382) & (hist < hist.shift(1))).astype(float)

        cols[f"own_fib236_net_{tf_name}"] = trig_bull_236 - trig_bear_236
        bull_rr = ((0.618 - f_pos).clip(lower=0.0)) / f_pos.clip(lower=0.05)
        bear_rr = ((f_pos - 0.382).clip(lower=0.0)) / (1.0 - f_pos).clip(lower=0.05)
        cols[f"own_fib382_rr_net_{tf_name}"] = trig_bull_236 * bull_rr - trig_bear_236 * bear_rr

    at_1h_sr = ((cols["own_fib_pos_1h"] <= 0.25) | (cols["own_fib_pos_1h"] >= 0.75)).astype(float)

    # =========================================================================
    # BLOCK B: Coin's OWN V4 Proportional Magnitude-Gap Divergences (Rules R1 - R4)
    # =========================================================================
    hi12, lo12 = h.rolling(12, min_periods=3).max(), l.rolling(12, min_periods=3).min()
    f_pos12 = (c - lo12) / ((hi12 - lo12).clip(lower=1e-8))
    bull_hook12 = ((f_pos12.rolling(4).min() <= 0.18) & (f_pos12 >= 0.236) & (hist > hist.shift(1))).astype(float)
    bear_hook12 = ((f_pos12.rolling(4).max() >= 0.82) & (f_pos12 <= 0.764) & (hist < hist.shift(1))).astype(float)

    # Rule R1: DIF/RSI/Vol vs Price Standardized Magnitude Gap (both hooked at 1h S/R and continuous)
    for w in (6, 12, 18, 36, 72):
        dz_p = _roll_z((c - c.shift(w)) / (atr14 * np.sqrt(w)), 288)
        dz_d = _roll_z(dif - dif.shift(w), 288)
        dz_r = _roll_z(rsi14 - rsi14.shift(w), 288)
        dz_v = _roll_z(vol_z - vol_z.shift(w), 288)

        gap_d = dz_d - 0.85 * dz_p
        gap_r = dz_r - 0.85 * dz_p
        gap_v = -np.sign(dz_p) * dz_v  # High volume without price progress or price move on low volume

        hooked_d = np.where(gap_d > 0.5, gap_d * bull_hook12, np.where(gap_d < -0.5, gap_d * bear_hook12, 0.0))
        hooked_r = np.where(gap_r > 0.5, gap_r * bull_hook12, np.where(gap_r < -0.5, gap_r * bear_hook12, 0.0))

        s_hd = pd.Series(hooked_d, index=idx).ewm(span=3, adjust=False).mean()
        s_hr = pd.Series(hooked_r, index=idx).ewm(span=3, adjust=False).mean()

        cols[f"own_r1_dif_sr_w{w}"] = s_hd * at_1h_sr
        cols[f"own_r1_dif_pure_w{w}"] = s_hd * (1.0 - at_1h_sr)
        cols[f"own_r1_rsi_sr_w{w}"] = s_hr * at_1h_sr
        cols[f"own_r1_dif_cont_w{w}"] = _soft_gate(gap_d, tau=1.15)
        cols[f"own_r1_rsi_cont_w{w}"] = _soft_gate(gap_r, tau=1.15)
        cols[f"own_r1_vol_gap_w{w}"] = _soft_gate(gap_v, tau=1.15)

    # Rule R2: Coin's OWN Order-Flow Effort vs. Price Result Absorption Gap (w = 2, 3, 6, 12, 24)
    for w in (2, 3, 6, 12, 24):
        zp = _roll_z(np.log(c / c.shift(w)), 288)
        zo_own = _roll_z(ofi_coin.rolling(w, min_periods=1).mean(), 288)
        zo_mkt = _roll_z(ofi_4c_al.rolling(w, min_periods=1).mean(), 288)
        cols[f"own_r2_abs_own_w{w}"] = _soft_gate(0.65 * zp - zo_own, tau=1.15)
        cols[f"own_r2_abs_mkt_w{w}"] = _soft_gate(0.65 * zp - zo_mkt, tau=1.15)

    # Rule R3: Coin's OWN Trade-Velocity vs. Price Magnitude Gap (w = 3, 6, 12)
    for w in (3, 6, 12):
        zp = _roll_z(np.log(c / c.shift(w)), 288)
        scnt = np.sign(zp) * _roll_z(np.log(cnt.rolling(w, min_periods=1).mean() + 1.0), 288)
        cols[f"own_r3_cnt_lead_w{w}"] = _soft_gate(scnt - 0.70 * zp, tau=1.15)

    # Rule R4: Coin's OWN DIF/RSI Channel Position vs. Price Channel Position Discrepancy (w = 18, 36, 72)
    for w in (18, 36, 72):
        p_pos = (c - l.rolling(w, min_periods=6).min()) / ((h.rolling(w, min_periods=6).max() - l.rolling(w, min_periods=6).min()).clip(lower=1e-8))
        d_pos = (dif - dif.rolling(w, min_periods=6).min()) / ((dif.rolling(w, min_periods=6).max() - dif.rolling(w, min_periods=6).min()).clip(lower=1e-8))
        r_pos = (rsi14 - rsi14.rolling(w, min_periods=6).min()) / ((rsi14.rolling(w, min_periods=6).max() - rsi14.rolling(w, min_periods=6).min()).clip(lower=1e-8))
        cols[f"own_r4_dif_chan_w{w}"] = _soft_gate(_roll_z(d_pos - p_pos, 288), tau=1.0)
        cols[f"own_r4_rsi_chan_w{w}"] = _soft_gate(_roll_z(r_pos - p_pos, 288), tau=1.0)

    # =========================================================================
    # BLOCK C: Coin's OWN MACD Near-Cross Rejection ("将穿未穿") & Slight-Cross Snapback
    # =========================================================================
    hist_abs = hist.abs()
    rel_hist = hist_abs / (hist_abs.rolling(36, min_periods=6).max() + 1e-12)
    same_sign = (np.sign(hist) == np.sign(hist.shift(3))) & (np.sign(hist) == np.sign(hist.shift(1)))
    cols["own_macd_near_cross_reject"] = np.where(
        same_sign & (rel_hist.shift(2) <= 0.22) & (hist_abs > hist_abs.shift(1)), np.sign(hist), 0.0
    )
    shallow_cross = (rel_hist.rolling(4, min_periods=1).max().shift(1) <= 0.28) & (np.sign(hist) != np.sign(hist.shift(2)))
    cols["own_macd_slight_cross_snapback"] = np.where(shallow_cross & (hist_abs > hist_abs.shift(1)), np.sign(hist), 0.0)
    cols["own_vw_clv_w12"] = (clv_coin * qv).rolling(12, min_periods=1).sum() / (qv.rolling(12, min_periods=1).sum() + 1e-12)

    # =========================================================================
    # BLOCK D: Cross-Asset Target-vs-BTC Relative Magnitude-Gap Divergence
    # =========================================================================
    c_btc = df_btc["close"].astype(float).reindex(idx).ffill()
    for w in (2, 3, 6, 12):
        zp_c = _roll_z(np.log(c / c.shift(w)), 288)
        zp_b = _roll_z(np.log(c_btc / c_btc.shift(w)), 288)
        cols[f"own_r5_btc_vs_coin_gap_w{w}"] = _soft_gate(zp_b - 0.85 * zp_c, tau=1.0)

    return pd.DataFrame(cols, index=idx).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
