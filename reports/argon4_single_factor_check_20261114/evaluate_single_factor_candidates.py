"""Reproducible Single-Factor Controlled Evaluation for ARGON-4 v15r1 Guard Candidates.

Evaluates two isolated single-factor candidates without modifying production policy or runtime state:
  1. Candidate 1 (Price Floor Only): `min_buy_price` + all 6 `stream_min_buy_price` `0.47 -> 0.445`,
     separating local `account.sqlite3` evidence from optional `--cloud-snapshot` evidence, with
     canonical UTC ISO timestamp normalization and `signal_id -> cohort_id` join for 100%
     `policy_version` attribution.
  2. Candidate 2 (US Open Blackout End Only): Strict 1-parameter control comparing online
     `policy_A` (`us_open_blackout.local_end_minute = 720`, exact unmodified `v15r1` policy) vs
     `policy_B` (`copy.deepcopy(policy_A)` with ONLY `us_open_blackout.local_end_minute = 600`)
     via `model_v14.apply_volatility_regime_guard` on strict 2026 OOS (`LOCAL_V4_FROZEN`,
     trained <= 2025-12-31). Zero extra non-online filters are applied in the primary A/B table;
     offline price/handicap/rank slices are reported strictly in a secondary diagnostic section.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any
from zoneinfo import ZoneInfo

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


def canon_utc_iso(ts: Any) -> str | None:
    """Normalize both `...Z` and `...+00:00` ISO strings to canonical UTC `+00:00` format."""
    if ts is None:
        return None
    txt = str(ts).strip()
    if not txt:
        return None
    return pd.Timestamp(txt).tz_convert("UTC").isoformat()


def polymarket_taker_fee_per_share(price: float, rate: float = 0.07) -> float:
    return float(rate) * float(price) * (1.0 - float(price))


def load_local_sqlite_evidence(runtime_dir: Path) -> dict[str, Any]:
    acc_path = runtime_dir / "account.sqlite3"
    bk_path = runtime_dir / "book_telemetry.sqlite3"
    out: dict[str, Any] = {
        "orders": [],
        "skips": [],
        "cohorts_by_utc": {},
        "signal_to_policy": {},
        "book_telemetry_summary": None,
    }
    if acc_path.is_file():
        con = sqlite3.connect(f"file:{acc_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        resolutions = {
            str(r["market_id"]): json.loads(r["payload"])
            for r in con.execute("SELECT market_id, payload FROM resolutions")
        }
        cohorts_by_utc: dict[str, str] = {}
        for r in con.execute("SELECT cohort_id, policy_version FROM decision_cohorts"):
            c_iso = canon_utc_iso(r["cohort_id"])
            if c_iso and r["policy_version"]:
                cohorts_by_utc[c_iso] = str(r["policy_version"])
        out["cohorts_by_utc"] = cohorts_by_utc

        signal_to_policy: dict[str, str] = {}
        for r in con.execute("SELECT cohort_id, signal_id, receipt_json FROM cohort_streams"):
            c_iso = canon_utc_iso(r["cohort_id"])
            pol_ver = cohorts_by_utc.get(c_iso or "")
            if not pol_ver:
                continue
            if r["signal_id"]:
                signal_to_policy[str(r["signal_id"])] = pol_ver
            if r["receipt_json"]:
                try:
                    rec = json.loads(r["receipt_json"])
                    sig_id = (rec.get("signal") or {}).get("signal_id")
                    if sig_id:
                        signal_to_policy[str(sig_id)] = pol_ver
                except Exception:
                    pass
        out["signal_to_policy"] = signal_to_policy

        order_to_policy: dict[str, str] = {
            str(r["order_id"]): str(r["policy_version"])
            for r in con.execute(
                "SELECT ac.order_id, dc.policy_version "
                "FROM allocation_consumption ac "
                "JOIN decision_cohorts dc ON ac.plan_id = dc.plan_id"
            )
            if r["order_id"] and r["policy_version"]
        }

        for r in con.execute(
            "SELECT order_id, signal_id, market_id, underlying, outcome, quantity, cost_atoms, fee_atoms, status, fill_at, payload "
            "FROM orders ORDER BY fill_at"
        ):
            p = json.loads(r["payload"])
            ws_raw = (p.get("venue_market") or {}).get("window_start")
            ws = canon_utc_iso(ws_raw)
            pol_ver = (
                signal_to_policy.get(str(r["signal_id"]))
                or order_to_policy.get(str(r["order_id"]))
                or (cohorts_by_utc.get(ws) if ws else None)
            )
            qty = float(r["quantity"])
            cost = r["cost_atoms"] / 1e6
            fee = r["fee_atoms"] / 1e6
            out["orders"].append(
                {
                    "source_host": "local",
                    "order_id": str(r["order_id"]),
                    "signal_id": str(r["signal_id"]),
                    "market_id": str(r["market_id"]),
                    "underlying": r["underlying"],
                    "window_start": ws,
                    "policy_version": pol_ver,
                    "outcome": r["outcome"],
                    "status": r["status"],
                    "quantity": qty,
                    "cost_usd": cost,
                    "fee_usd": fee,
                    "avg_price": (cost / qty) if qty > 0 else None,
                    "resolution": resolutions.get(str(r["market_id"])),
                }
            )
        for r in con.execute(
            "SELECT cohort_id, stream, rank, state, receipt_json FROM cohort_streams "
            "WHERE state IN ('PRICE_LIMIT_NOT_MET', 'EXECUTION_FRESHNESS_VETO') ORDER BY cohort_id"
        ):
            rec = json.loads(r["receipt_json"])
            sig = rec.get("signal") or {}
            book = rec.get("orderbook") or {}
            asks = sorted(
                [
                    (float(a["price"]), float(a["size"]))
                    for a in (book.get("asks") or [])
                    if float(a.get("size", 0)) > 0
                ]
            )
            mid = str(sig.get("market_id") or "")
            c_iso = canon_utc_iso(r["cohort_id"])
            pol_ver = (
                signal_to_policy.get(str(sig.get("signal_id") or ""))
                or (cohorts_by_utc.get(c_iso) if c_iso else None)
            )
            out["skips"].append(
                {
                    "source_host": "local",
                    "cohort_id": c_iso,
                    "policy_version": pol_ver,
                    "stream": r["stream"],
                    "rank": r["rank"],
                    "state": r["state"],
                    "veto_reason": rec.get("veto_reason"),
                    "market_id": mid,
                    "outcome": sig.get("outcome"),
                    "limit_price": rec.get("limit_price", 0.515),
                    "min_buy_price": rec.get("min_buy_price", 0.0),
                    "best_ask": asks[0][0] if asks else rec.get("exec_best_ask"),
                    "top5_asks": asks[:5],
                    "fee_rate": (rec.get("fee_quote") or {}).get("rate"),
                    "resolution": resolutions.get(mid),
                }
            )
        con.close()

    if bk_path.is_file():
        bcon = sqlite3.connect(f"file:{bk_path}?mode=ro", uri=True)
        et = ZoneInfo("America/New_York")
        bk_stats = {
            "et_10_12_weekday": {"n": 0, "in_047_049": 0, "in_047_050": 0, "in_0445_047": 0, "depth25_le050": 0},
            "other_hours": {"n": 0, "in_047_049": 0, "in_047_050": 0, "in_0445_047": 0, "depth25_le050": 0},
        }
        for cohort_s, _off, status_s, pjson in bcon.execute(
            "SELECT cohort, target_offset, status, payload_json FROM samples WHERE target_offset IN (6, 10)"
        ):
            if status_s in ("FAILED_OBSERVATION", "INVALID_OBSERVATION", "MISSED_SAMPLE", "PLANNED", "DISPATCHED"):
                continue
            row = json.loads(pjson)
            pbook = row.get("parsed_book") or {}
            asks = [
                (float(a["price"]), float(a["size"]))
                for a in (pbook.get("asks") or [])
                if float(a.get("size", 0)) > 0
            ]
            if not asks:
                continue
            asks.sort()
            best = asks[0][0]
            dt_et = pd.Timestamp(cohort_s).tz_convert(et)
            mins = dt_et.hour * 60 + dt_et.minute
            bkt = bk_stats["et_10_12_weekday"] if (dt_et.weekday() < 5 and 600 <= mins < 720) else bk_stats["other_hours"]
            bkt["n"] += 1
            if 0.47 <= best <= 0.49:
                bkt["in_047_049"] += 1
            if 0.47 <= best <= 0.50:
                bkt["in_047_050"] += 1
            if 0.445 <= best < 0.47:
                bkt["in_0445_047"] += 1
            dep = sum(p * sz * (1.0 + polymarket_taker_fee_per_share(p, 0.07) / p) for p, sz in asks if 0.47 <= p <= 0.50)
            if dep >= 25.0:
                bkt["depth25_le050"] += 1
        bcon.close()
        out["book_telemetry_summary"] = bk_stats
    return out


def normalize_external_snapshot(
    snap: dict[str, Any],
    fallback_cohorts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Normalize timestamps and backfill missing `policy_version` on an exported cloud snapshot JSON."""
    cohorts_map: dict[str, str] = dict(fallback_cohorts or {})
    for s in snap.get("skips", []):
        c_iso = canon_utc_iso(s.get("cohort_id"))
        if c_iso and s.get("policy_version"):
            cohorts_map[c_iso] = str(s["policy_version"])
    for o in snap.get("orders", []):
        ws_iso = canon_utc_iso(o.get("window_start"))
        if ws_iso and o.get("policy_version"):
            cohorts_map[ws_iso] = str(o["policy_version"])

    norm_orders = []
    for o in snap.get("orders", []):
        ws_iso = canon_utc_iso(o.get("window_start"))
        pv = o.get("policy_version") or (cohorts_map.get(ws_iso) if ws_iso else None)
        norm_orders.append({**o, "window_start": ws_iso, "policy_version": pv})

    norm_skips = []
    for s in snap.get("skips", []):
        c_iso = canon_utc_iso(s.get("cohort_id"))
        pv = s.get("policy_version") or (cohorts_map.get(c_iso) if c_iso else None)
        norm_skips.append({**s, "cohort_id": c_iso, "policy_version": pv})

    return {**snap, "orders": norm_orders, "skips": norm_skips}


def infer_gamma_winner(gamma_obj: dict[str, Any] | None) -> str | None:
    if not isinstance(gamma_obj, dict):
        return None
    if gamma_obj.get("closed") is not True or str(gamma_obj.get("umaResolutionStatus") or "").lower() != "resolved":
        return None
    outs = gamma_obj.get("outcomes") or []
    prs = gamma_obj.get("outcomePrices") or []
    if len(outs) != 2 or len(prs) != 2:
        return None
    for lbl, pr in zip(outs, prs):
        if abs(float(pr) - 1.0) < 1e-6:
            return str(lbl).strip().upper()
    return None


def _summarize_single_evidence_source(
    ev: dict[str, Any],
    source_label: str,
    current_policy_version: str,
) -> dict[str, Any]:
    settled_orders = [
        o
        for o in ev.get("orders", [])
        if o.get("status") in ("WON", "LOST") and o.get("avg_price") is not None
    ]
    unattributed_orders = sum(1 for o in settled_orders if not o.get("policy_version"))

    bands = [
        ("<0.445", 0.0, 0.445),
        ("[0.445, 0.470)", 0.445, 0.470),
        ("[0.470, 0.490]", 0.470, 0.4900001),
        ("(0.490, 0.515]", 0.4900001, 0.5150001),
    ]
    band_rows = []
    for label, lo, hi in bands:
        sub = [o for o in settled_orders if lo <= float(o["avg_price"]) < hi]
        n = len(sub)
        n_v15r1 = sum(1 for o in sub if o.get("policy_version") == current_policy_version)
        wins = sum(1 for o in sub if o["status"] == "WON")
        spent = sum(float(o["cost_usd"]) + float(o["fee_usd"]) for o in sub)
        payout = sum(float(o["quantity"]) if o["status"] == "WON" else 0.0 for o in sub)
        pnl = payout - spent
        band_rows.append(
            {
                "price_band": label,
                "filled_n": n,
                "current_v15r1_n": n_v15r1,
                "pre_v15r1_n": n - n_v15r1,
                "wins": wins,
                "win_rate_pct": round(wins / n * 100.0, 2) if n else None,
                "total_spent_usd": round(spent, 2),
                "realized_pnl_usd": round(pnl, 2),
                "roi_pct": round(pnl / spent * 100.0, 2) if spent > 0 else None,
            }
        )

    floor_skips = []
    for s in ev.get("skips", []):
        ba = s.get("best_ask")
        min_bp = float(s.get("min_buy_price") or 0.0)
        if ba is None or s.get("state") != "PRICE_LIMIT_NOT_MET":
            continue
        if not (0.445 <= float(ba) < 0.470 and min_bp >= 0.470 - 1e-9):
            continue
        winner = None
        if isinstance(s.get("resolution"), dict):
            winner = str(s["resolution"].get("outcome") or "").upper() or None
        if winner is None:
            winner = infer_gamma_winner(s.get("gamma_official"))
        floor_skips.append(
            {
                "source_host": source_label,
                "cohort_id": s.get("cohort_id"),
                "policy_version": s.get("policy_version"),
                "is_current_v15r1": s.get("policy_version") == current_policy_version,
                "stream": s.get("stream"),
                "outcome": s.get("outcome"),
                "best_ask": float(ba),
                "min_buy_price": min_bp,
                "official_winner": winner,
                "would_have_won": (winner == str(s.get("outcome") or "").upper()) if winner in ("UP", "DOWN") else None,
            }
        )

    v15r1_floor_skips = [s for s in floor_skips if s["is_current_v15r1"]]
    fills_0445_0470 = [
        {
            "window_start": o.get("window_start"),
            "policy_version": o.get("policy_version"),
            "is_current_v15r1": o.get("policy_version") == current_policy_version,
            "underlying": o.get("underlying"),
            "outcome": o.get("outcome"),
            "avg_price": round(float(o["avg_price"]), 4),
            "status": o.get("status"),
            "spent_usd": round(float(o["cost_usd"]) + float(o["fee_usd"]), 2),
            "pnl_usd": round(
                (float(o["quantity"]) if o["status"] == "WON" else 0.0)
                - float(o["cost_usd"])
                - float(o["fee_usd"]),
                2,
            ),
        }
        for o in sorted(settled_orders, key=lambda x: str(x.get("window_start") or ""))
        if 0.445 <= float(o["avg_price"]) < 0.470
    ]

    return {
        "source": source_label,
        "current_policy_version": current_policy_version,
        "settled_orders_total": len(settled_orders),
        "unattributed_policy_version_orders": unattributed_orders,
        "settled_paper_orders_by_band": band_rows,
        "filled_orders_in_0445_0470_detail": fills_0445_0470,
        "skipped_by_047_floor_in_0445_0470_all": floor_skips,
        "current_v15r1_floor_rejected_in_0445_0470_count": len(v15r1_floor_skips),
        "sample_sufficiency_verdict": (
            "INSUFFICIENT_CURRENT_V15R1_FLOOR_REJECT_SAMPLES"
            if len(v15r1_floor_skips) < 30
            else "SUFFICIENT_FOR_EVALUATION"
        ),
    }


def evaluate_candidate_1_min_buy_0445(
    local_ev: dict[str, Any],
    current_policy_version: str,
    cloud_ev: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Candidate 1: Isolate real Polymarket orders and 0.47-floor skips in [0.445, 0.470),
    keeping local `account.sqlite3` strictly separated from optional `--cloud-snapshot` data."""
    res: dict[str, Any] = {
        "local_only": _summarize_single_evidence_source(local_ev, "local", current_policy_version),
    }
    if cloud_ev is not None:
        norm_cloud = normalize_external_snapshot(cloud_ev, fallback_cohorts=local_ev.get("cohorts_by_utc"))
        res["optional_cloud_snapshot"] = _summarize_single_evidence_source(
            norm_cloud, "cloud", current_policy_version
        )
    return res


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
            },
            index=df.index,
        )
    return out


def _opt_finite(val: float) -> float | None:
    return float(val) if math.isfinite(val) else None


def evaluate_candidate_2_us_open_1000_et(
    oos_dir: Path,
    data_root: Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Candidate 2: Strict 1-parameter A/B comparison on 2026 OOS (`LOCAL_V4_FROZEN`, trained <= 2025-12-31).

    Primary comparison calls `model_v14.apply_volatility_regime_guard` directly with:
      - `guard_A`: unmodified `policy["volatility_regime_guard"]` (`us_open_blackout.local_end_minute == 720`)
      - `guard_B`: `deepcopy(guard_A)` with ONLY `us_open_blackout.local_end_minute = 600` changed.
    Zero extra filters (`handicap_bps >= -1.0`, approximate slot limits, etc.) are applied to the primary table.
    """
    ex_ante = compute_ex_ante_series_5m(data_root)
    parts = []
    for st in ACTIVE_STREAMS:
        p = pd.read_parquet(oos_dir / f"{st}_oos_predictions.parquet")
        p = p[p["variant"] == "LOCAL_V4_FROZEN"].copy()
        parts.append(p)
    df = pd.concat(parts, ignore_index=True)
    df["cohort_utc"] = pd.to_datetime(df["decision_close_utc"], utc=True)
    total_days = (df["cohort_utc"].max() - df["cohort_utc"].min()).total_seconds() / 86400.0

    # Filter only to raw model signals (`selected == True`, or `stream_thresholds` if configured in policy)
    stream_th = policy.get("stream_thresholds") if isinstance(policy.get("stream_thresholds"), dict) else {}
    if stream_th:
        eff_th = df["stream"].map(stream_th).fillna(df["threshold"]).to_numpy(dtype=np.float64)
        raw_sel_mask = np.abs(df["score"].to_numpy(dtype=np.float64)) >= eff_th
    else:
        eff_th = df["threshold"].to_numpy(dtype=np.float64)
        raw_sel_mask = df["selected"].to_numpy(dtype=bool)

    sig_df = df.loc[raw_sel_mask].copy().reset_index(drop=True)
    sig_df["eff_threshold"] = eff_th[raw_sel_mask]

    btc = ex_ante["BTCUSDT"].reindex(sig_df["cohort_utc"])
    sig_df["btc_ret_1h_pct"] = btc["ret_1h_pct"].to_numpy()
    sig_df["btc_ret_2h_pct"] = btc["ret_2h_pct"].to_numpy()
    sig_df["btc_ret_4h_pct"] = btc["ret_4h_pct"].to_numpy()
    sig_df["btc_ema_gap_bps"] = btc["ema_gap_bps"].to_numpy()
    sig_df["btc_bar1_range_bps"] = btc["bar1_range_bps"].to_numpy()
    sig_df["btc_vol_surge_24"] = btc["vol_surge_24"].to_numpy()

    n_sig = len(sig_df)
    rv12 = np.empty(n_sig, dtype=np.float64)
    own_15m = np.empty(n_sig, dtype=np.float64)
    own_1h = np.empty(n_sig, dtype=np.float64)
    own_s3 = np.empty(n_sig, dtype=np.int32)
    tw_t = np.empty(n_sig, dtype=np.float64)
    sp_t = np.empty(n_sig, dtype=np.float64)
    tw_ret = np.empty(n_sig, dtype=np.float64)

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

    sig_df["rv12_bps"] = rv12
    sig_df["own_ret_15m_bps"] = own_15m
    sig_df["own_ret_1h_pct"] = own_1h
    sig_df["own_streak_3bar"] = own_s3
    sig_df["tw60_t"] = tw_t
    sig_df["spot_close_t"] = sp_t
    sig_df["twap_ret"] = tw_ret
    sgn = np.where(sig_df["side"].to_numpy() == "UP", 1.0, -1.0)
    sig_df["handicap_bps"] = ((tw_t - sp_t) / sp_t * 10000.0) * sgn
    sig_df["twap_settled"] = np.isfinite(tw_ret) & np.isfinite(sig_df["handicap_bps"])
    sig_df["twap_proxy_win"] = np.where(sig_df["side"] == "UP", tw_ret >= 0.0, tw_ret < 0.0)
    sig_df["perp_point_win"] = sig_df["win"].astype(bool)
    sig_df["both_labels_settled"] = (
        sig_df["settled"].to_numpy(dtype=bool)
        & (~sig_df["label_tie"].to_numpy(dtype=bool))
        & sig_df["twap_settled"].to_numpy(dtype=bool)
    )
    sig_df["month"] = sig_df["cohort_utc"].dt.strftime("%Y-%m")

    guard_A = deepcopy(policy["volatility_regime_guard"])
    guard_B = deepcopy(policy["volatility_regime_guard"])
    assert int(guard_A["us_open_blackout"]["local_end_minute"]) == 720
    guard_B["us_open_blackout"]["local_end_minute"] = 600

    pol_max_cap = float(policy.get("max_buy_price", 0.525))

    sel_A = np.zeros(n_sig, dtype=bool)
    sel_B = np.zeros(n_sig, dtype=bool)
    reason_A: list[str | None] = [None] * n_sig
    cap_A = np.full(n_sig, pol_max_cap, dtype=np.float64)
    cap_B = np.full(n_sig, pol_max_cap, dtype=np.float64)

    streams_arr = sig_df["stream"].to_numpy()
    sides_arr = sig_df["side"].to_numpy()
    scores_arr = sig_df["score"].to_numpy(dtype=np.float64)
    base_th_arr = sig_df["threshold"].to_numpy(dtype=np.float64)
    eff_th_arr = sig_df["eff_threshold"].to_numpy(dtype=np.float64)
    cohorts_arr = sig_df["cohort_utc"].tolist()
    btc_1h_arr = sig_df["btc_ret_1h_pct"].to_numpy(dtype=np.float64)
    btc_2h_arr = sig_df["btc_ret_2h_pct"].to_numpy(dtype=np.float64)
    btc_4h_arr = sig_df["btc_ret_4h_pct"].to_numpy(dtype=np.float64)
    btc_gap_arr = sig_df["btc_ema_gap_bps"].to_numpy(dtype=np.float64)
    btc_vs_arr = sig_df["btc_vol_surge_24"].to_numpy(dtype=np.float64)
    btc_rng_arr = sig_df["btc_bar1_range_bps"].to_numpy(dtype=np.float64)

    for i in range(n_sig):
        base_pred = {
            "selected": True,
            "side": str(sides_arr[i]),
            "z_v4": float(scores_arr[i]),
            "threshold_frozen": float(eff_th_arr[i]),
            "base_threshold_frozen": float(base_th_arr[i]),
        }
        s_tw = _opt_finite(tw_t[i])
        s_sc = _opt_finite(sp_t[i])
        tw_unavail = s_tw is None or s_sc is None
        common_kwargs = {
            "ex_ante_rv12_bps": _opt_finite(rv12[i]),
            "cohort_time": cohorts_arr[i],
            "btc_ret_1h_pct": _opt_finite(btc_1h_arr[i]),
            "btc_ret_2h_pct": _opt_finite(btc_2h_arr[i]),
            "btc_ret_4h_pct": _opt_finite(btc_4h_arr[i]),
            "btc_ema_gap_bps": _opt_finite(btc_gap_arr[i]),
            "btc_vol_surge_24": _opt_finite(btc_vs_arr[i]),
            "btc_bar1_range_bps": _opt_finite(btc_rng_arr[i]),
            "own_ret_1h_pct": _opt_finite(own_1h[i]),
            "own_ret_15m_bps": _opt_finite(own_15m[i]),
            "own_streak_3bar": int(own_s3[i]),
            "spot_twap60_t": s_tw,
            "spot_close_t": s_sc,
            "twap60_unavailable": tw_unavail,
        }
        out_A = model_v14.apply_volatility_regime_guard(
            str(streams_arr[i]), base_pred, guard_config=guard_A, **common_kwargs
        )
        out_B = model_v14.apply_volatility_regime_guard(
            str(streams_arr[i]), base_pred, guard_config=guard_B, **common_kwargs
        )
        sel_A[i] = bool(out_A.get("selected"))
        sel_B[i] = bool(out_B.get("selected"))
        reason_A[i] = out_A.get("volatility_regime_reason")
        if out_A.get("effective_max_buy_price") is not None:
            cap_A[i] = min(pol_max_cap, float(out_A["effective_max_buy_price"]))
        if out_B.get("effective_max_buy_price") is not None:
            cap_B[i] = min(pol_max_cap, float(out_B["effective_max_buy_price"]))

    # Strict monotonicity & single-parameter diff invariants:
    assert int(((~sel_B) & sel_A).sum()) == 0, "Relaxing local_end_minute 720->600 cannot reject baseline signals"
    unlocked_mask = sel_B & (~sel_A)
    for idx_u in np.flatnonzero(unlocked_mask):
        assert reason_A[int(idx_u)] == "US_OPEN_BLACKOUT", f"Unexpected guard reason in diff: {reason_A[int(idx_u)]}"

    sig_df["sel_A"] = sel_A
    sig_df["sel_B"] = sel_B
    sig_df["cap_A"] = cap_A
    sig_df["cap_B"] = cap_B

    settled_mask = sig_df["both_labels_settled"].to_numpy(dtype=bool)
    sub_A = sig_df.loc[sel_A & settled_mask].copy()
    sub_B = sig_df.loc[sel_B & settled_mask].copy()
    sub_diff = sig_df.loc[unlocked_mask & settled_mask].copy()

    def summarize_slice(sub: pd.DataFrame, cap_col: str = "cap_B", include_breakdowns: bool = False) -> dict[str, Any]:
        n = int(len(sub))
        if n == 0:
            return {"n": 0}
        perp_wr = float(sub["perp_point_win"].mean())
        twap_wr = float(sub["twap_proxy_win"].mean())

        caps = sub[cap_col].to_numpy(dtype=np.float64)
        unit_costs_at_cap = caps + 0.07 * caps * (1.0 - caps)
        perp_wins_f = sub["perp_point_win"].to_numpy(dtype=np.float64)
        twap_wins_f = sub["twap_proxy_win"].to_numpy(dtype=np.float64)
        worst_cap_spent = float(unit_costs_at_cap.sum())
        worst_cap_roi_perp = (float(perp_wins_f.sum()) / worst_cap_spent - 1.0) * 100.0
        worst_cap_roi_twap = (float(twap_wins_f.sum()) / worst_cap_spent - 1.0) * 100.0

        tier_counts = {
            f"{k:.3f}": int(v)
            for k, v in sorted(sub[cap_col].round(4).value_counts().items())
        }

        sensitivity = []
        for p_fill in (0.470, 0.480, 0.485, 0.490, 0.498, 0.505):
            unit_cost = p_fill + polymarket_taker_fee_per_share(p_fill, 0.07)
            sensitivity.append(
                {
                    "assumed_fill_price": p_fill,
                    "breakeven_wr_pct": round(unit_cost * 100.0, 2),
                    "ev_roi_perp_pct": round((perp_wr / unit_cost - 1.0) * 100.0, 2),
                    "ev_roi_twap_proxy_pct": round((twap_wr / unit_cost - 1.0) * 100.0, 2),
                }
            )

        out_s: dict[str, Any] = {
            "n": n,
            "signals_per_calendar_day": round(n / total_days, 3),
            "perp_point_wr_pct": round(perp_wr * 100.0, 2),
            "twap_1m_ohlc4_proxy_wr_pct": round(twap_wr * 100.0, 2),
            "effective_max_buy_price_distribution": tier_counts,
            "worst_case_fill_at_each_signal_cap_roi_perp_pct": round(worst_cap_roi_perp, 2),
            "worst_case_fill_at_each_signal_cap_roi_twap_proxy_pct": round(worst_cap_roi_twap, 2),
            "fixed_fill_price_sensitivity": sensitivity,
        }
        if include_breakdowns:
            by_month = []
            for m, grp in sub.groupby("month"):
                by_month.append(
                    {
                        "month": m,
                        "n": int(len(grp)),
                        "perp_wr_pct": round(float(grp["perp_point_win"].mean()) * 100.0, 2),
                        "twap_proxy_wr_pct": round(float(grp["twap_proxy_win"].mean()) * 100.0, 2),
                    }
                )
            by_stream = []
            for st_name in ACTIVE_STREAMS:
                grp = sub[sub["stream"] == st_name]
                if len(grp) == 0:
                    continue
                by_stream.append(
                    {
                        "stream": st_name,
                        "n": int(len(grp)),
                        "perp_wr_pct": round(float(grp["perp_point_win"].mean()) * 100.0, 2),
                        "twap_proxy_wr_pct": round(float(grp["twap_proxy_win"].mean()) * 100.0, 2),
                    }
                )
            out_s["by_month"] = by_month
            out_s["by_stream"] = by_stream
        return out_s

    # Secondary non-online sensitivity slices on `sub_diff` (`B \ A`), strictly separated from primary control:
    rank_map = {s: idx for idx, s in enumerate(policy["order"])}
    sub_diff_ranked = sub_diff.assign(wf_rank=sub_diff["stream"].map(rank_map)).sort_values(["cohort_utc", "wf_rank"])
    sub_diff_rank1 = sub_diff_ranked.groupby("cohort_utc", as_index=False).first()
    sub_diff_hcap_ge_neg1 = sub_diff[sub_diff["handicap_bps"] >= -1.0].copy()

    return {
        "oos_span_utc": [str(df["cohort_utc"].min()), str(df["cohort_utc"].max())],
        "calendar_days": round(total_days, 2),
        "changed_parameter_only": {
            "path": "volatility_regime_guard.us_open_blackout.local_end_minute",
            "baseline_A": 720,
            "candidate_B": 600,
            "extra_filters_applied": False,
        },
        "strict_single_parameter_ab": {
            "policy_A_baseline_end_720": summarize_slice(sub_A, cap_col="cap_A", include_breakdowns=False),
            "policy_B_candidate_end_600": summarize_slice(sub_B, cap_col="cap_B", include_breakdowns=False),
            "newly_unlocked_delta_B_minus_A": summarize_slice(sub_diff, cap_col="cap_B", include_breakdowns=True),
        },
        "secondary_non_online_diagnostics_on_unlocked_delta": {
            "disclaimer": (
                "Diagnostic slices below are NOT part of online argon4_forward_policy.json. "
                "Historical 2026 OOS files lack Polymarket orderbook asks/depths; actual fill rates "
                "and waterfall budget consumption require live/shadow book telemetry."
            ),
            "rank1_per_cohort_lower_bound": summarize_slice(sub_diff_rank1, cap_col="cap_B", include_breakdowns=False),
            "offline_proxy_handicap_ge_neg1_bps": summarize_slice(
                sub_diff_hcap_ge_neg1, cap_col="cap_B", include_breakdowns=False
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=ROOT / "paper_runtime" / "argon4_forward_20261105",
        help="Local paper_runtime directory containing account.sqlite3 and book_telemetry.sqlite3",
    )
    parser.add_argument(
        "--cloud-snapshot",
        type=Path,
        default=None,
        help="Optional read-only JSON export from cloud account.sqlite3 / book_telemetry.sqlite3",
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
        "--policy",
        type=Path,
        default=ROOT / "config" / "argon4_forward_policy.json",
    )
    args = parser.parse_args()

    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    current_ver = str(policy.get("version") or "")
    local_ev = load_local_sqlite_evidence(args.runtime_dir)
    cloud_ev = (
        json.loads(args.cloud_snapshot.read_text(encoding="utf-8"))
        if args.cloud_snapshot and args.cloud_snapshot.is_file()
        else None
    )

    c1 = evaluate_candidate_1_min_buy_0445(local_ev, current_ver, cloud_ev)
    c2 = evaluate_candidate_2_us_open_1000_et(args.oos_dir, args.data_root, policy)

    report = {
        "candidate_1_min_buy_047_to_0445": c1,
        "candidate_2_us_open_end_1200_to_1000_et": c2,
        "observed_book_telemetry_10_12_et": {
            "local": local_ev.get("book_telemetry_summary"),
            **(
                {"optional_cloud_snapshot": cloud_ev.get("book_telemetry_summary")}
                if cloud_ev and cloud_ev.get("book_telemetry_summary")
                else {}
            ),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
