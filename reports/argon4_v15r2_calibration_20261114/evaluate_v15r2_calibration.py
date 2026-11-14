"""Reproducible Multi-Parameter Calibration & Ablation Audit for ARGON-4 v15r2-Calibrated.

Documents and deterministically reproduces the step-by-step transition from `v15r1`
(`reports/argon4_single_factor_check_20261114/baseline_policy_v15r1.py`, `commit bb2eacd`)
to `v15r2-Calibrated` (`config/argon4_forward_policy.json`), auditing every modified parameter:
  1. Tick-aligned confidence buy-price tiers (`0.49/0.50`, `0.50/0.51` @ `dz>=0.25`,
     `0.51/0.52` @ `dz>=0.60`, `BTC_15m` top `0.52` @ `dz>=1.0`, `counter_15m_marginal: 0.49`),
     fixing `$0.01`-tick floor truncation of `0.498 -> 0.49` and `0.502 -> 0.50`.
  2. Two-sided `twap60_handicap_guard` lower bound (`min_5m_handicap_bps = -0.5`,
     `min_15m_handicap_bps = -0.5`) to block toxic stale-quote discounts on deep-ITM starts.
  3. Causal `twap60_handicap_guard` upper-bound calibration (`max_5m: 1.0 -> 2.2 bps`,
     `max_15m: 2.0 -> 4.0 bps`), resolving the `v15r1` `[0.47, 0.49]` price-cap vs `+1.0/+2.0 bps`
     handicap deadlock without over-relaxing into `[+4.5, +7.0] bps` adverse selection.
  4. Horizon-differentiated `stream_min_buy_price` (`15m: 0.47 -> 0.46`, `5m` kept at `0.47`;
     `0.45` rejected due to 2026 causal negative EV; `us_open_blackout` kept at `12:00 ET`).

Provenance & Governance Note (`AGENTS.md`):
  - Real market metrics (`real_market_guard_selection`) use only observed Binance 5m Futures &
    1m Spot OHLC4 series (`twap_1m_ohlc4_proxy_wr_pct`, `perp_point_wr_pct`) with zero synthetic
    candles or fills.
  - Because historical Polymarket L2 orderbooks do not exist across full 2024-2026 backtests,
    orderbook fill-band filtering (`causal_t0_tick_book_simulation` vs `post_t_1m_bar_leaky_diagnostic`)
    is explicitly isolated and labeled as an analytical simulation diagnostic, never represented
    as historical exchange fills or live deployment authorization.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.argon4_forward import model_v14

ACTIVE_STREAMS = ["BTC_15m", "ETH_5m", "ETH_15m", "BTC_5m", "SOL_15m", "SOL_5m"]
STREAM_SYM = {
    "BTC_15m": ("BTCUSDT", 3),
    "BTC_5m": ("BTCUSDT", 1),
    "ETH_15m": ("ETHUSDT", 3),
    "ETH_5m": ("ETHUSDT", 1),
    "SOL_15m": ("SOLUSDT", 3),
    "SOL_5m": ("SOLUSDT", 1),
}


def _std_norm_cdf(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import ndtr  # type: ignore[import-untyped]

        return np.asarray(ndtr(x), dtype=np.float64)
    except Exception:
        erf_vec = np.vectorize(math.erf, otypes=[np.float64])
        return 0.5 * (1.0 + erf_vec(np.asarray(x, dtype=np.float64) / math.sqrt(2.0)))


def polymarket_net_pnl_per_100(win: np.ndarray, ask: np.ndarray, fee_rate: float = 0.07) -> np.ndarray:
    fee_per_share = fee_rate * ask * (1.0 - ask)
    eff_cost = ask + fee_per_share
    return np.where(win, 100.0 * (1.0 / eff_cost - 1.0), -100.0)


def compute_ex_ante_series_5m(data_root: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        fpath = data_root / "intraday" / f"{sym}_5m_futures_2023_2026.parquet"
        df = pd.read_parquet(fpath)
        idx = pd.to_datetime(df["open_time"], utc=True) if "open_time" in df.columns else pd.to_datetime(df.index, utc=True)
        df = df.copy()
        df.index = idx + pd.Timedelta(minutes=5)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        o = df["open"].astype(np.float64)
        h = df["high"].astype(np.float64)
        l = df["low"].astype(np.float64)
        c = df["close"].astype(np.float64)
        qv = df["quote_volume"].astype(np.float64)

        body_bps = (c - o).abs() / o * 10000.0
        rv12_bps = body_bps.rolling(12, min_periods=12).mean()
        ret_15m_bps = (c / o.shift(2) - 1.0) * 10000.0
        ret_1h_pct = (c / o.shift(11) - 1.0) * 100.0
        ret_2h_pct = (c / o.shift(23) - 1.0) * 100.0
        ret_4h_pct = (c / o.shift(47) - 1.0) * 100.0
        up3 = ((c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2))).astype(np.int32) * 3
        dn3 = ((c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2))).astype(np.int32) * (-3)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema36 = c.ewm(span=36, adjust=False).mean()
        ema_gap_bps = (ema12 / ema36 - 1.0) * 10000.0
        bar1_range_bps = (h - l) / o * 10000.0
        vol_surge_24 = qv / qv.shift(1).rolling(24, min_periods=24).median()

        spath = data_root / "spot_1m" / f"{sym}_1m_spot.parquet"
        s1 = pd.read_parquet(spath)
        s1["bound_ts"] = pd.to_datetime(s1["bound_ts"], utc=True)
        s1 = s1.drop_duplicates("bound_ts", keep="last").set_index("bound_ts").sort_index()
        tw60_t = s1["twap60"].reindex(df.index).astype(np.float64)
        spot_close_t = s1["spot_close"].reindex(df.index).astype(np.float64)
        tw60_end_5m = s1["twap60"].reindex(df.index + pd.Timedelta(minutes=5)).astype(np.float64)
        tw60_end_15m = s1["twap60"].reindex(df.index + pd.Timedelta(minutes=15)).astype(np.float64)
        tw60_end_5m.index = df.index
        tw60_end_15m.index = df.index

        out[sym] = pd.DataFrame(
            {
                "rv12_bps": rv12_bps,
                "ret_15m_bps": ret_15m_bps,
                "ret_1h_pct": ret_1h_pct,
                "ret_2h_pct": ret_2h_pct,
                "ret_4h_pct": ret_4h_pct,
                "streak_3bar": up3 + dn3,
                "ema_gap_bps": ema_gap_bps,
                "bar1_range_bps": bar1_range_bps,
                "vol_surge_24": vol_surge_24,
                "tw60_t": tw60_t,
                "spot_close_t": spot_close_t,
                "twap_ret_5m": tw60_end_5m / tw60_t - 1.0,
                "twap_ret_15m": tw60_end_15m / tw60_t - 1.0,
                "s2t_ret_5m": tw60_end_5m / spot_close_t - 1.0,
                "s2t_ret_15m": tw60_end_15m / spot_close_t - 1.0,
            },
            index=df.index,
        )
    return out


def _opt_finite(val: float) -> float | None:
    return float(val) if math.isfinite(val) else None


def build_ablation_policies(v15r1_policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Construct the exact step-by-step cumulative policy ladder from v15r1 -> v15r2-Calibrated."""
    p0 = deepcopy(v15r1_policy)

    # Step 1: Align confidence buy-price tiers to Polymarket $0.01 tick grid
    p1 = deepcopy(p0)
    ct = p1["volatility_regime_guard"]["confidence_tiered_max_buy_price"]
    ct["tiers"] = [
        {"min_delta_z": 0.0, "max_5m": 0.49, "max_15m": 0.50},
        {"min_delta_z": 0.25, "max_5m": 0.50, "max_15m": 0.51},
        {"min_delta_z": 0.60, "max_5m": 0.51, "max_15m": 0.52},
    ]
    ct["stream_top_tiers"] = {"BTC_15m": {"min_delta_z": 1.0, "max_buy_price": 0.525}}
    ct["counter_15m_marginal_max_buy_price"] = 0.49

    # Step 2: Add two-sided lower bound (-0.5 bps) on twap60_handicap_guard
    p2 = deepcopy(p1)
    p2["volatility_regime_guard"]["twap60_handicap_guard"]["min_handicap_bps"] = -0.5

    # Step 3: Relax twap60_handicap_guard upper bound to causal optimum (+2.2 / +4.0 bps)
    p3 = deepcopy(p2)
    p3["volatility_regime_guard"]["twap60_handicap_guard"]["max_5m_handicap_bps"] = 2.2
    p3["volatility_regime_guard"]["twap60_handicap_guard"]["max_15m_handicap_bps"] = 4.0

    # Step 4 (Production v15r2-Calibrated): Lower 15m stream_min_buy_price 0.47 -> 0.46 (keep 5m at 0.47)
    p4 = deepcopy(p3)
    p4["min_buy_price"] = 0.46
    for st in ("BTC_15m", "ETH_15m", "SOL_15m", "BNB_15m"):
        if st in p4.get("stream_min_buy_price", {}):
            p4["stream_min_buy_price"][st] = 0.46
    for st in ("BTC_5m", "ETH_5m", "SOL_5m"):
        if st in p4.get("stream_min_buy_price", {}):
            p4["stream_min_buy_price"][st] = 0.47

    # Rejected comparison A: Over-wide handicap (+4.5 / +7.0 bps) + 15m floor 0.45
    p_alt_wide = deepcopy(p4)
    p_alt_wide["min_buy_price"] = 0.45
    for st in ("BTC_15m", "ETH_15m", "SOL_15m", "BNB_15m"):
        if st in p_alt_wide.get("stream_min_buy_price", {}):
            p_alt_wide["stream_min_buy_price"][st] = 0.45
    p_alt_wide["volatility_regime_guard"]["twap60_handicap_guard"]["max_5m_handicap_bps"] = 4.5
    p_alt_wide["volatility_regime_guard"]["twap60_handicap_guard"]["max_15m_handicap_bps"] = 7.0

    # Rejected comparison B: Also shortening us_open_blackout 12:00 ET -> 10:30 ET
    p_alt_us1030 = deepcopy(p_alt_wide)
    p_alt_us1030["volatility_regime_guard"]["us_open_blackout"]["local_end_minute"] = 630

    return {
        "step0_v15r1_baseline": p0,
        "step1_plus_tick_aligned_tiers": p1,
        "step2_plus_min_handicap_neg05": p2,
        "step3_plus_causal_handicap_22_40": p3,
        "step4_v15r2_calibrated_production": p4,
        "rejected_alt_wide_45_70_floor045": p_alt_wide,
        "rejected_alt_wide_plus_us_open_1030": p_alt_us1030,
    }


def verify_exact_match_with_head_policy(step4_policy: dict[str, Any], head_policy: dict[str, Any]) -> dict[str, Any]:
    """Assert that Step 4 matches every active guard/tier/price-floor field in config/argon4_forward_policy.json."""
    checked = {
        "min_buy_price": (step4_policy["min_buy_price"], head_policy["min_buy_price"]),
        "max_buy_price": (step4_policy["max_buy_price"], head_policy["max_buy_price"]),
        "stream_min_buy_price": (step4_policy["stream_min_buy_price"], head_policy["stream_min_buy_price"]),
        "twap60_handicap_guard": (
            {
                k: step4_policy["volatility_regime_guard"]["twap60_handicap_guard"][k]
                for k in ("enabled", "min_handicap_bps", "max_5m_handicap_bps", "max_15m_handicap_bps")
            },
            {
                k: head_policy["volatility_regime_guard"]["twap60_handicap_guard"][k]
                for k in ("enabled", "min_handicap_bps", "max_5m_handicap_bps", "max_15m_handicap_bps")
            },
        ),
        "confidence_tiered_max_buy_price": (
            {
                k: step4_policy["volatility_regime_guard"]["confidence_tiered_max_buy_price"][k]
                for k in ("enabled", "tiers", "stream_top_tiers", "counter_15m_marginal_max_buy_price")
            },
            {
                k: head_policy["volatility_regime_guard"]["confidence_tiered_max_buy_price"][k]
                for k in ("enabled", "tiers", "stream_top_tiers", "counter_15m_marginal_max_buy_price")
            },
        ),
        "us_open_blackout_local_end_minute": (
            step4_policy["volatility_regime_guard"]["us_open_blackout"]["local_end_minute"],
            head_policy["volatility_regime_guard"]["us_open_blackout"]["local_end_minute"],
        ),
    }
    mismatches = {k: {"step4": v[0], "head": v[1]} for k, v in checked.items() if v[0] != v[1]}
    assert not mismatches, f"Policy calibration drift detected vs head policy: {mismatches}"
    return {"verified_identical_to_head_policy": True, "head_policy_version": head_policy.get("version")}


def evaluate_local_2026_oos_ladder(
    oos_dir: Path,
    data_root: Path,
    policies: dict[str, dict[str, Any]],
    seed: int = 42,
) -> dict[str, Any]:
    ex_ante = compute_ex_ante_series_5m(data_root)
    parts = []
    for st in ACTIVE_STREAMS:
        p = pd.read_parquet(oos_dir / f"{st}_oos_predictions.parquet")
        p = p[p["variant"] == "LOCAL_V4_FROZEN"].copy()
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    df["cohort_utc"] = pd.to_datetime(df["decision_close_utc"], utc=True)
    sig_df = df.loc[df["selected"].to_numpy(dtype=bool)].copy().reset_index(drop=True)
    total_days = (sig_df["cohort_utc"].max() - sig_df["cohort_utc"].min()).total_seconds() / 86400.0

    btc = ex_ante["BTCUSDT"].reindex(sig_df["cohort_utc"])
    for col_src, col_dst in [
        ("ret_1h_pct", "btc_ret_1h_pct"),
        ("ret_2h_pct", "btc_ret_2h_pct"),
        ("ret_4h_pct", "btc_ret_4h_pct"),
        ("ema_gap_bps", "btc_ema_gap_bps"),
        ("bar1_range_bps", "btc_bar1_range_bps"),
        ("vol_surge_24", "btc_vol_surge_24"),
    ]:
        sig_df[col_dst] = btc[col_src].to_numpy()

    n_sig = len(sig_df)
    rv12 = np.empty(n_sig, dtype=np.float64)
    own_15m = np.empty(n_sig, dtype=np.float64)
    own_1h = np.empty(n_sig, dtype=np.float64)
    own_s3 = np.empty(n_sig, dtype=np.int32)
    tw_t = np.empty(n_sig, dtype=np.float64)
    sp_t = np.empty(n_sig, dtype=np.float64)
    tw_ret = np.empty(n_sig, dtype=np.float64)
    s2t_ret = np.empty(n_sig, dtype=np.float64)

    for st, idxs in sig_df.groupby("stream").groups.items():
        sym, k = STREAM_SYM[st]
        sub_ex = ex_ante[sym].reindex(sig_df.loc[idxs, "cohort_utc"])
        pos = sig_df.index.get_indexer(idxs)
        rv12[pos] = sub_ex["rv12_bps"].to_numpy()
        own_15m[pos] = sub_ex["ret_15m_bps"].to_numpy()
        own_1h[pos] = sub_ex["ret_1h_pct"].to_numpy()
        own_s3[pos] = sub_ex["streak_3bar"].fillna(0).to_numpy(dtype=np.int32)
        tw_t[pos] = sub_ex["tw60_t"].to_numpy()
        sp_t[pos] = sub_ex["spot_close_t"].to_numpy()
        tw_ret[pos] = sub_ex["twap_ret_15m" if k == 3 else "twap_ret_5m"].to_numpy()
        s2t_ret[pos] = sub_ex["s2t_ret_15m" if k == 3 else "s2t_ret_5m"].to_numpy()

    is_5m = sig_df["stream"].str.endswith("_5m").to_numpy(dtype=bool)
    side_up = (sig_df["side"].to_numpy() == "UP")
    sgn = np.where(side_up, 1.0, -1.0)
    hcap_bps = ((tw_t - sp_t) / sp_t * 10000.0) * sgn
    valid = (
        sig_df["settled"].to_numpy(dtype=bool)
        & (~sig_df["label_tie"].to_numpy(dtype=bool))
        & np.isfinite(tw_ret)
        & np.isfinite(hcap_bps)
    )
    twap_win = np.where(side_up, tw_ret >= 0.0, tw_ret < 0.0)
    s2t_win = np.where(side_up, s2t_ret >= 0.0, s2t_ret < 0.0)
    perp_win = sig_df["win"].to_numpy(dtype=bool)
    months = sig_df["cohort_utc"].dt.strftime("%Y-%m").to_numpy()

    # Analytical $0.01-tick causal T+0s ask vs [T, T+1m) post-T leaky ask diagnostic
    rng = np.random.default_rng(seed)
    rv_c = np.clip(np.where(np.isfinite(rv12), rv12, 8.0), 4.5, 80.0)
    sigma_h = rv_c * np.where(is_5m, 0.92, math.sqrt(3.0) * 0.96)
    fair_p = _std_norm_cdf(-np.where(np.isfinite(hcap_bps), hcap_bps, 0.0) / sigma_h)
    half_spread = np.where(is_5m, 0.012, 0.015)
    indep_noise = rng.normal(0.0, 0.014, size=n_sig)
    ask_causal = np.clip(np.round(fair_p + half_spread + indep_noise, 2), 0.05, 0.95)
    ask_leaky = np.clip(np.round(fair_p + half_spread + indep_noise + 0.018 * np.where(s2t_win, 1.0, -1.0), 2), 0.05, 0.95)

    streams_arr = sig_df["stream"].to_numpy()
    sides_arr = sig_df["side"].to_numpy()
    scores_arr = sig_df["score"].to_numpy(dtype=np.float64)
    th_arr = sig_df["threshold"].to_numpy(dtype=np.float64)
    cohorts_list = sig_df["cohort_utc"].tolist()
    btc_1h_arr = sig_df["btc_ret_1h_pct"].to_numpy(dtype=np.float64)
    btc_2h_arr = sig_df["btc_ret_2h_pct"].to_numpy(dtype=np.float64)
    btc_4h_arr = sig_df["btc_ret_4h_pct"].to_numpy(dtype=np.float64)
    btc_gap_arr = sig_df["btc_ema_gap_bps"].to_numpy(dtype=np.float64)
    btc_vs_arr = sig_df["btc_vol_surge_24"].to_numpy(dtype=np.float64)
    btc_rng_arr = sig_df["btc_bar1_range_bps"].to_numpy(dtype=np.float64)

    ladder_results: dict[str, Any] = {}
    for name, pol in policies.items():
        guard_cfg = pol["volatility_regime_guard"]
        pol_max = float(pol.get("max_buy_price", 0.525))
        pol_min = float(pol.get("min_buy_price", 0.47))
        st_min_map = pol.get("stream_min_buy_price") or {}

        sel = np.zeros(n_sig, dtype=bool)
        eff_cap = np.full(n_sig, pol_max, dtype=np.float64)
        eff_floor = np.array([float(st_min_map.get(str(s), pol_min)) for s in streams_arr], dtype=np.float64)

        for i in range(n_sig):
            if not valid[i]:
                continue
            base_pred = {
                "selected": True,
                "side": str(sides_arr[i]),
                "z_v4": float(scores_arr[i]),
                "threshold_frozen": float(th_arr[i]),
                "base_threshold_frozen": float(th_arr[i]),
            }
            s_tw = _opt_finite(tw_t[i])
            s_sc = _opt_finite(sp_t[i])
            out_g = model_v14.apply_volatility_regime_guard(
                str(streams_arr[i]),
                base_pred,
                guard_config=guard_cfg,
                ex_ante_rv12_bps=_opt_finite(rv12[i]),
                cohort_time=cohorts_list[i],
                btc_ret_1h_pct=_opt_finite(btc_1h_arr[i]),
                btc_ret_2h_pct=_opt_finite(btc_2h_arr[i]),
                btc_ret_4h_pct=_opt_finite(btc_4h_arr[i]),
                btc_ema_gap_bps=_opt_finite(btc_gap_arr[i]),
                btc_vol_surge_24=_opt_finite(btc_vs_arr[i]),
                btc_bar1_range_bps=_opt_finite(btc_rng_arr[i]),
                own_ret_1h_pct=_opt_finite(own_1h[i]),
                own_ret_15m_bps=_opt_finite(own_15m[i]),
                own_streak_3bar=int(own_s3[i]),
                spot_twap60_t=s_tw,
                spot_close_t=s_sc,
                twap60_unavailable=(s_tw is None or s_sc is None),
            )
            sel[i] = bool(out_g.get("selected"))
            if out_g.get("effective_max_buy_price") is not None:
                eff_cap[i] = min(pol_max, float(out_g["effective_max_buy_price"]))

        # Tick-floor effective cap on $0.01 grid (matching online CLOB limit behavior)
        tick_cap = np.floor(eff_cap * 100.0 + 1e-9) / 100.0
        n_sel = int(sel.sum())
        real_summary = {
            "n_guard_passed": n_sel,
            "signals_per_day": round(n_sel / total_days, 2),
            "twap_1m_ohlc4_proxy_wr_pct": round(100.0 * float(twap_win[sel].mean()), 2) if n_sel else 0.0,
            "perp_point_wr_pct": round(100.0 * float(perp_win[sel].mean()), 2) if n_sel else 0.0,
            "effective_tick_cap_distribution": {
                f"{k:.2f}": int(v) for k, v in sorted(pd.Series(tick_cap[sel]).value_counts().items())
            },
        }

        def _sim_book(ask_arr: np.ndarray) -> dict[str, Any]:
            m_fill = sel & (ask_arr >= eff_floor - 1e-9) & (ask_arr <= tick_cap + 1e-9)
            nf = int(m_fill.sum())
            if nf == 0:
                return {"n_filled": 0}
            pnl = polymarket_net_pnl_per_100(twap_win[m_fill], ask_arr[m_fill])
            mo_f = months[m_fill]
            f1 = float(pnl[mo_f <= "2026-04"].sum())
            f2 = float(pnl[(mo_f >= "2026-05") & (mo_f <= "2026-08")].sum())
            f3 = float(pnl[mo_f >= "2026-09"].sum())
            return {
                "n_filled": nf,
                "fill_rate_of_raw_signals_pct": round(100.0 * nf / int(valid.sum()), 2),
                "avg_ask": round(float(ask_arr[m_fill].mean()), 3),
                "twap_proxy_wr_pct": round(100.0 * float(twap_win[m_fill].mean()), 2),
                "ev_per_100_usd": round(float(pnl.mean()), 2),
                "net_pnl_100_stake_usd": round(float(pnl.sum()), 0),
                "folds_net_pnl_usd": {"F1_Jan_Apr": round(f1, 0), "F2_May_Aug": round(f2, 0), "F3_Sep": round(f3, 0)},
            }

        ladder_results[name] = {
            "real_market_guard_selection": real_summary,
            "causal_t0_tick_book_simulation": _sim_book(ask_causal),
            "post_t_1m_bar_leaky_diagnostic": _sim_book(ask_leaky),
        }

    return {
        "dataset": "LOCAL_V4_FROZEN strict 2026 OOS (trained <= 2025-12-31)",
        "span_utc": [str(sig_df["cohort_utc"].min()), str(sig_df["cohort_utc"].max())],
        "calendar_days": round(total_days, 2),
        "raw_settled_signals": int(valid.sum()),
        "ablation_ladder": ladder_results,
    }


def evaluate_pure_walkforward_30m_parquet(parquet_path: Path, seed: int = 42) -> dict[str, Any]:
    """Evaluate the 30-month (2024-05 .. 2026-11) strictly out-of-sample `SPOT_TO_TWAP60_PURE`
    walk-forward signal table (`pure_signals_2024_2026.parquet`, 123,893 raw signals) when supplied."""
    df = pd.read_parquet(parquet_path)
    df = df[df["stream"].isin(ACTIVE_STREAMS)].copy()
    df = df[np.isfinite(df["handicap_bps"])].reset_index(drop=True)

    stream = df["stream"].to_numpy()
    is_5m = df["is_5m"].to_numpy(dtype=bool)
    side_up = df["side_up"].to_numpy(dtype=bool)
    rv12 = df["rv12_bps"].to_numpy(dtype=np.float64)
    ny_wd = df["ny_wd"].to_numpy()
    ny_min = df["ny_min"].to_numpy()
    utc_wd = df["utc_wd"].to_numpy()
    is_hol = df["is_ny_hol"].to_numpy(dtype=bool)
    btc_1h = df["btc_ret_1h_pct"].to_numpy(dtype=np.float64)
    btc_4h = df["btc_ret_4h_pct"].to_numpy(dtype=np.float64)
    own_1h = df["own_ret_1h_pct"].to_numpy(dtype=np.float64)
    own_15m = df["own_ret_15m_bps"].to_numpy(dtype=np.float64)
    dz = df["delta_z"].to_numpy(dtype=np.float64)
    hcap = df["handicap_bps"].to_numpy(dtype=np.float64)
    win_arr = df["twap_win"].to_numpy(dtype=bool)
    s2t_win = df["s2t_win"].to_numpy(dtype=bool)
    yr = df["year_int"].to_numpy()
    mo = df["month_str"].to_numpy()

    min_rv = np.select(
        [stream == "BTC_5m", stream == "BTC_15m", stream == "ETH_5m", stream == "ETH_15m", stream == "SOL_5m", stream == "SOL_15m"],
        [3.0, 2.6, 4.5, 3.6, 6.5, 5.5],
        default=3.0,
    )
    min_rv = np.where(is_5m & ((utc_wd >= 5) | is_hol), np.maximum(min_rv, 4.0), min_rv)
    ok_rv = np.isfinite(rv12) & (rv12 >= min_rv)
    blk_830 = (ny_wd < 5) & (ny_min >= 505) & (ny_min < 540)
    blk_sun = (ny_wd == 6) & (((ny_min >= 1080) & (ny_min < 1200)) | ((ny_min >= 1260) & (ny_min < 1320)))
    is_be = np.isin(stream, ["BTC_15m", "BTC_5m", "ETH_15m", "ETH_5m"])
    blk_4h = is_be & (np.abs(btc_4h) > 2.0)
    blk_chase = is_be & np.where(side_up, btc_1h > 0.30, btc_1h < -0.30)
    lim_1h = np.select([np.char.startswith(stream.astype(str), "BTC"), np.char.startswith(stream.astype(str), "ETH")], [3.0, 3.5], default=4.0)
    lim_15m = np.select([np.char.startswith(stream.astype(str), "BTC"), np.char.startswith(stream.astype(str), "ETH")], [200.0, 220.0], default=250.0)
    blk_crash = np.where(side_up, own_1h < -lim_1h, own_1h > lim_1h) | np.where(side_up, own_15m < -lim_15m, own_15m > lim_15m)
    base_keep = ok_rv & (~blk_830) & (~blk_sun) & (~blk_4h) & (~blk_chase) & (~blk_crash)

    rng = np.random.default_rng(seed)
    rv_c = np.clip(rv12, 4.5, 80.0)
    sigma_h = rv_c * np.where(is_5m, 0.92, math.sqrt(3.0) * 0.96)
    fair_p = _std_norm_cdf(-hcap / sigma_h)
    half_spread = np.where(is_5m, 0.012, 0.015)
    indep_noise = rng.normal(0.0, 0.014, size=len(df))
    ask_causal = np.clip(np.round(fair_p + half_spread + indep_noise, 2), 0.05, 0.95)
    ask_leaky = np.clip(np.round(fair_p + half_spread + indep_noise + 0.018 * np.where(s2t_win, 1.0, -1.0), 2), 0.05, 0.95)

    counter_45 = np.where(side_up, own_15m <= -45.0, own_15m >= 45.0)
    cap_v15r1 = np.where(is_5m, 0.49, 0.498)
    cap_v15r1 = np.where(dz >= 0.35, np.where(is_5m, 0.502, 0.51), cap_v15r1)
    cap_v15r1 = np.where(dz >= 0.80, np.where(is_5m, 0.51, 0.52), cap_v15r1)
    cap_v15r1 = np.where((stream == "BTC_15m") & (dz >= 1.0), 0.525, cap_v15r1)
    cap_v15r1 = np.floor(np.where(counter_45 & (dz < 0.12), np.minimum(cap_v15r1, 0.475), cap_v15r1) * 100.0 + 1e-9) / 100.0

    cap_v15r2 = np.where(is_5m, 0.49, 0.50)
    cap_v15r2 = np.where(dz >= 0.25, np.where(is_5m, 0.50, 0.51), cap_v15r2)
    cap_v15r2 = np.where(dz >= 0.60, np.where(is_5m, 0.51, 0.52), cap_v15r2)
    cap_v15r2 = np.where((stream == "BTC_15m") & (dz >= 1.0), 0.525, cap_v15r2)
    cap_v15r2 = np.floor(np.where(counter_45 & (dz < 0.12), np.minimum(cap_v15r2, 0.49), cap_v15r2) * 100.0 + 1e-9) / 100.0

    specs = {
        "step0_v15r1_baseline": (-999.0, 1.0, 2.0, 720, cap_v15r1, 0.47, 0.47),
        "step1_plus_tick_aligned_tiers": (-999.0, 1.0, 2.0, 720, cap_v15r2, 0.47, 0.47),
        "step2_plus_min_handicap_neg05": (-0.5, 1.0, 2.0, 720, cap_v15r2, 0.47, 0.47),
        "step3_plus_causal_handicap_22_40": (-0.5, 2.2, 4.0, 720, cap_v15r2, 0.47, 0.47),
        "step4_v15r2_calibrated_production": (-0.5, 2.2, 4.0, 720, cap_v15r2, 0.47, 0.46),
        "rejected_alt_wide_45_70_floor045": (-0.5, 4.5, 7.0, 720, cap_v15r2, 0.47, 0.45),
        "rejected_alt_wide_plus_us_open_1030": (-0.5, 4.5, 7.0, 630, cap_v15r2, 0.47, 0.45),
    }

    ladder_30m: dict[str, Any] = {}
    for name, (h_min, h_max5, h_max15, us_end, cap_arr, fl5, fl15) in specs.items():
        h_ok = np.where(is_5m, (hcap >= h_min) & (hcap <= h_max5), (hcap >= h_min) & (hcap <= h_max15))
        us_blk = (ny_wd < 5) & (ny_min >= 570) & (ny_min < us_end)
        fl_arr = np.where(is_5m, fl5, fl15)
        keep = base_keep & h_ok & (~us_blk) & (ask_causal >= fl_arr - 1e-9) & (ask_causal <= cap_arr + 1e-9)
        sub_w = win_arr[keep]
        sub_a = ask_causal[keep]
        sub_p = polymarket_net_pnl_per_100(sub_w, sub_a)
        sub_y = yr[keep]
        sub_m = mo[keep]

        def _blk(mask: np.ndarray) -> dict[str, Any]:
            n = int(mask.sum())
            return {
                "n": n,
                "wr_pct": round(100.0 * float(sub_w[mask].mean()), 2) if n else 0.0,
                "avg_ask": round(float(sub_a[mask].mean()), 3) if n else 0.0,
                "ev_100_usd": round(float(sub_p[mask].mean()), 2) if n else 0.0,
                "net_pnl_usd": round(float(sub_p[mask].sum()), 0) if n else 0.0,
            }

        pos_2425 = sum(1 for _, g in pd.Series(sub_p[sub_y < 2026]).groupby(sub_m[sub_y < 2026]) if g.sum() > 0)
        tot_2425 = len(np.unique(sub_m[sub_y < 2026]))
        pos_26 = sum(1 for _, g in pd.Series(sub_p[sub_y == 2026]).groupby(sub_m[sub_y == 2026]) if g.sum() > 0)
        tot_26 = len(np.unique(sub_m[sub_y == 2026]))

        ladder_30m[name] = {
            "2024_2025_18m": {**_blk(sub_y < 2026), "positive_months": f"{pos_2425}/{tot_2425}"},
            "2026_11m": {**_blk(sub_y == 2026), "positive_months": f"{pos_26}/{tot_26}"},
            "2026_folds_net_pnl_usd": {
                "F1_Jan_Apr": _blk((sub_y == 2026) & (sub_m <= "2026-04"))["net_pnl_usd"],
                "F2_May_Aug": _blk((sub_y == 2026) & (sub_m >= "2026-05") & (sub_m <= "2026-08"))["net_pnl_usd"],
                "F3_Sep_Nov": _blk((sub_y == 2026) & (sub_m >= "2026-09"))["net_pnl_usd"],
            },
            "ALL_30m": _blk(np.ones(len(sub_p), dtype=bool)),
        }

    # Marginal 15m ask == 0.45 vs 0.46 slice explaining why 15m floor was stopped at 0.46 (not 0.45)
    m15_base = base_keep & (~((ny_wd < 5) & (ny_min >= 570) & (ny_min < 720))) & (~is_5m) & (hcap >= -0.5) & (hcap <= 4.0)
    floor_045_vs_046: dict[str, Any] = {}
    for a_val in (0.46, 0.45):
        for lbl, ym in [("2024_2025", yr < 2026), ("2026", yr == 2026)]:
            m = m15_base & ym & np.isclose(ask_causal, a_val)
            p_arr = polymarket_net_pnl_per_100(win_arr[m], ask_causal[m])
            floor_045_vs_046[f"ask_{a_val:.2f}_{lbl}"] = {
                "n": int(m.sum()),
                "wr_pct": round(100.0 * float(win_arr[m].mean()), 2),
                "ev_100_usd": round(float(p_arr.mean()), 2),
                "net_pnl_usd": round(float(p_arr.sum()), 0),
            }

    # Causal T+0s vs post-T 1m-bar leaky bucket comparison on 2026 executable band [0.46, 0.51]
    p26_base = (yr == 2026) & base_keep & (~((ny_wd < 5) & (ny_min >= 570) & (ny_min < 720)))
    bucket_comp: dict[str, dict[str, Any]] = {"CAUSAL_T0": {}, "POST_T_1M_BAR_LEAKY": {}}
    for b_lbl, a_src in [("CAUSAL_T0", ask_causal), ("POST_T_1M_BAR_LEAKY", ask_leaky)]:
        for lo, hi in [(-10.0, 0.0), (-0.5, 0.0), (0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.5), (4.5, 7.0)]:
            m = p26_base & (hcap > lo) & (hcap <= hi) & (a_src >= 0.46) & (a_src <= 0.51)
            if int(m.sum()) > 0:
                p_arr = polymarket_net_pnl_per_100(win_arr[m], a_src[m])
                bucket_comp[b_lbl][f"({lo}, {hi}]"] = {
                    "n": int(m.sum()),
                    "avg_ask": round(float(a_src[m].mean()), 3),
                    "twap_wr_pct": round(100.0 * float(win_arr[m].mean()), 2),
                    "ev_100_usd": round(float(p_arr.mean()), 2),
                    "net_pnl_usd": round(float(p_arr.sum()), 0),
                }

    return {
        "dataset": "30-month strict walk-forward OOS SPOT_TO_TWAP60_PURE (2024-05 .. 2026-11-13)",
        "raw_signals": int(len(df)),
        "ablation_ladder_causal_t0": ladder_30m,
        "marginal_15m_floor_046_vs_045": floor_045_vs_046,
        "handicap_bucket_causal_vs_leaky_2026": bucket_comp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v15r1-policy",
        type=Path,
        default=None,
        help="Optional v15r1 baseline JSON path (defaults to pinned snapshot in baseline_policy_v15r1.py).",
    )
    parser.add_argument(
        "--head-policy",
        type=Path,
        default=ROOT / "config" / "argon4_forward_policy.json",
    )
    parser.add_argument(
        "--oos-dir",
        type=Path,
        default=ROOT / "reports" / "argon4_local_v4_20261106",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "research_data" / "argon4_twap60_20261113" / "raw_data",
    )
    parser.add_argument(
        "--pure-signals-parquet",
        type=Path,
        default=None,
        help="Optional 30-month SPOT_TO_TWAP60_PURE strict walk-forward OOS parquet for multi-year calibration summary.",
    )
    parser.add_argument(
        "--write-summary",
        type=Path,
        default=None,
        help="Optional path to write JSON summary.",
    )
    args = parser.parse_args()

    if args.v15r1_policy is not None:
        v15r1_pol = json.loads(args.v15r1_policy.read_text(encoding="utf-8"))
    else:
        from reports.argon4_single_factor_check_20261114.baseline_policy_v15r1 import get_v15r1_baseline_policy

        v15r1_pol = get_v15r1_baseline_policy()
    head_pol = json.loads(args.head_policy.read_text(encoding="utf-8"))
    policies = build_ablation_policies(v15r1_pol)
    parity = verify_exact_match_with_head_policy(policies["step4_v15r2_calibrated_production"], head_pol)
    local_2026 = evaluate_local_2026_oos_ladder(args.oos_dir, args.data_root, policies)

    report: dict[str, Any] = {
        "governance_disclaimer": (
            "Offline research & parameter-calibration reproduction helper only. "
            "Orderbook simulation blocks use analytical $0.01-tick models because full 2024-2026 "
            "historical Polymarket L2 books do not exist; does not represent live trading authorization (AGENTS.md)."
        ),
        "policy_parity_check": parity,
        "local_2026_strict_oos_evaluation": local_2026,
    }
    if args.pure_signals_parquet is not None and args.pure_signals_parquet.is_file():
        report["walkforward_30m_spot_to_twap60_pure_evaluation"] = evaluate_pure_walkforward_30m_parquet(
            args.pure_signals_parquet
        )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.write_summary is not None:
        args.write_summary.write_bytes(text.replace("\r\n", "\n").encode("utf-8") + b"\n")
    print(text)


if __name__ == "__main__":
    main()
