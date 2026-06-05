"""Per-CIK behavioural profiling with expanding-window stats.

Mirrors `analysis notebooks/cik_behavior_analysis.ipynb` but operates on
local parquet DataFrames (passed in) instead of Postgres, and supports an
"expanding window" cutoff so the profile only reflects data up to a given
(year, quarter) — no lookahead.

Public API:
  build_cik_profile_upto(year, quarter, norm_holdings, cik_aum) -> DataFrame
  tag_archetypes(profile) -> DataFrame (adds is_* flags)
  filter_ciks(profile, **kwargs) -> set of cik ids passing the filters
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _period(year: int, quarter: int) -> int:
    return int(year) * 4 + int(quarter) - 1


def build_cik_profile_upto(year, quarter, norm_holdings, cik_aum):
    """Aggregate per-CIK behavioural metrics using only rows whose
    period = year*4 + quarter - 1 is <= the target period.

    norm_holdings : DataFrame with columns [cik, cusip, year, quarter, weight]
    cik_aum       : DataFrame with columns [cik, year, quarter, total | aum]
    """
    cap = _period(year, quarter)

    nh = norm_holdings
    if "period" not in nh.columns:
        nh = nh.assign(period=nh["year"].astype(int) * 4
                              + nh["quarter"].astype(int) - 1)
    nh = nh[(nh["weight"].notna()) & (nh["weight"] > 0) & (nh["period"] <= cap)]
    nh = nh[["cik", "cusip", "period", "weight"]].reset_index(drop=True)

    aum_col = "total" if "total" in cik_aum.columns else "aum"
    aum = cik_aum.assign(period=cik_aum["year"].astype(int) * 4
                                + cik_aum["quarter"].astype(int) - 1)
    aum = aum[(aum[aum_col] > 0) & (aum["period"] <= cap)]
    aum = aum[["cik", "period", aum_col]].rename(columns={aum_col: "aum"})

    if len(nh) == 0:
        return pd.DataFrame(columns=[
            "cik", "n_quarters", "first_period", "last_period",
            "aum_mean", "aum_median", "n_holdings_mean", "n_holdings_std",
            "hhi_mean", "top5_weight_mean", "aum_log_std", "aum_cagr",
            "turnover_mean", "turnover_std", "open_rate_mean",
            "close_rate_mean", "avg_holding_duration",
        ])

    # n_holdings / HHI / top-5 weight per (cik, period)
    nh_sorted = nh.sort_values(["cik", "period", "weight"],
                               ascending=[True, True, False])
    nh_sorted["rk"] = nh_sorted.groupby(["cik", "period"]).cumcount() + 1
    nh_sorted["w_top5"] = np.where(nh_sorted["rk"] <= 5,
                                   nh_sorted["weight"], 0.0)
    nh_sorted["w_sq"] = nh_sorted["weight"] ** 2
    qstats = nh_sorted.groupby(["cik", "period"]).agg(
        n_holdings=("weight", "size"),
        hhi=("w_sq", "sum"),
        top5_weight=("w_top5", "sum"),
    ).reset_index()
    qstats = qstats.merge(aum, on=["cik", "period"], how="left")

    base = qstats.groupby("cik").agg(
        n_quarters=("period", "nunique"),
        first_period=("period", "min"),
        last_period=("period", "max"),
        aum_mean=("aum", "mean"),
        aum_median=("aum", "median"),
        n_holdings_mean=("n_holdings", "mean"),
        n_holdings_std=("n_holdings", "std"),
        hhi_mean=("hhi", "mean"),
        top5_weight_mean=("top5_weight", "mean"),
    )

    aum_obs = qstats.dropna(subset=["aum"]).sort_values(["cik", "period"]).copy()
    if len(aum_obs):
        aum_obs["log_aum"] = np.log(aum_obs["aum"])
        aum_log_std = aum_obs.groupby("cik")["log_aum"].std().rename("aum_log_std")
        first_last = aum_obs.groupby("cik").agg(
            aum_first=("aum", "first"),
            aum_last=("aum", "last"),
            p_first=("period", "first"),
            p_last=("period", "last"),
        )
        dt_years = ((first_last["p_last"] - first_last["p_first"]) / 4).clip(lower=0.25)
        first_last["aum_cagr"] = (
            (first_last["aum_last"] / first_last["aum_first"]) ** (1 / dt_years) - 1)
    else:
        aum_log_std = pd.Series(dtype=float, name="aum_log_std")
        first_last = pd.DataFrame(columns=["aum_cagr"])

    # turnover / open_rate / close_rate — pandas analogue of the notebook's
    # SQL self-union: pair each holding row with a shadow row at period+1.
    cur_rows = nh.rename(columns={"weight": "w_cur"}).assign(w_prev=0.0)
    prev_rows = nh.rename(columns={"weight": "w_prev"}).assign(w_cur=0.0)
    prev_rows["period"] = prev_rows["period"] + 1
    cols = ["cik", "cusip", "period", "w_cur", "w_prev"]
    joined = pd.concat([cur_rows[cols], prev_rows[cols]], ignore_index=True)
    joined = joined[joined["period"] <= cap + 1]
    joined = joined.groupby(["cik", "cusip", "period"], as_index=False).agg(
        w_cur=("w_cur", "sum"),
        w_prev=("w_prev", "sum"),
    )
    joined["abs_diff"] = (joined["w_cur"] - joined["w_prev"]).abs()
    joined["opened"] = ((joined["w_cur"] > 0) & (joined["w_prev"] == 0)).astype(int)
    joined["closed"] = ((joined["w_cur"] == 0) & (joined["w_prev"] > 0)).astype(int)
    joined["has_cur"] = (joined["w_cur"] > 0).astype(int)
    joined["has_prev"] = (joined["w_prev"] > 0).astype(int)
    ch_agg = joined.groupby(["cik", "period"]).agg(
        turnover=("abs_diff", "sum"),
        opened_n=("opened", "sum"),
        closed_n=("closed", "sum"),
        cur_n=("has_cur", "sum"),
        prev_n=("has_prev", "sum"),
    ).reset_index()
    ch_agg["turnover"] = 0.5 * ch_agg["turnover"]
    ch_agg["open_rate"] = ch_agg["opened_n"] / ch_agg["cur_n"].replace(0, np.nan)
    ch_agg["close_rate"] = ch_agg["closed_n"] / ch_agg["prev_n"].replace(0, np.nan)
    ch_agg = ch_agg[ch_agg["prev_n"] > 0]
    churn_agg = ch_agg.groupby("cik").agg(
        turnover_mean=("turnover", "mean"),
        turnover_std=("turnover", "std"),
        open_rate_mean=("open_rate", "mean"),
        close_rate_mean=("close_rate", "mean"),
    )

    # avg holding duration — gaps-and-islands on consecutive period runs
    nh_dur = nh.sort_values(["cik", "cusip", "period"]).copy()
    nh_dur["rn"] = nh_dur.groupby(["cik", "cusip"]).cumcount()
    nh_dur["run_key"] = nh_dur["period"] - nh_dur["rn"]
    run_len = (nh_dur.groupby(["cik", "cusip", "run_key"]).size()
                     .rename("run_len").reset_index())
    duration = run_len.groupby("cik")["run_len"].mean().rename("avg_holding_duration")

    out = (base
           .join(aum_log_std)
           .join(first_last[["aum_cagr"]] if "aum_cagr" in first_last.columns
                 else pd.DataFrame(columns=["aum_cagr"]))
           .join(churn_agg)
           .join(duration))
    return out.reset_index()


def tag_archetypes(profile: pd.DataFrame) -> pd.DataFrame:
    """Add boolean is_buy_and_hold / is_high_churn / is_concentrated /
    is_diversified flags using p25/p75 cutoffs of the profile (same rules
    as the notebook). Cutoffs are re-derived from the input — so expanding
    profiles get drifting thresholds, which is the intended behaviour."""
    q = profile.copy()
    if len(q) == 0:
        for c in ("is_buy_and_hold", "is_high_churn",
                  "is_concentrated", "is_diversified"):
            q[c] = False
        return q
    turnover_low  = q["turnover_mean"].quantile(0.25)
    turnover_high = q["turnover_mean"].quantile(0.75)
    duration_long = q["avg_holding_duration"].quantile(0.75)
    churn_high    = q["open_rate_mean"].quantile(0.75)
    conc_high     = q["hhi_mean"].quantile(0.75)
    conc_low      = q["hhi_mean"].quantile(0.25)
    q["is_buy_and_hold"] = ((q["turnover_mean"] <= turnover_low)
                            & (q["avg_holding_duration"] >= duration_long))
    q["is_high_churn"]   = ((q["turnover_mean"] >= turnover_high)
                            | (q["open_rate_mean"] >= churn_high))
    q["is_concentrated"] = q["hhi_mean"] >= conc_high
    q["is_diversified"]  = q["hhi_mean"] <= conc_low
    return q


_VALID_ARCHETYPES = ("buy_and_hold", "high_churn", "concentrated", "diversified")


def filter_ciks(
    profile: pd.DataFrame,
    *,
    archetype: str | None = None,
    min_turnover: float | None = None,
    max_turnover: float | None = None,
    max_open_rate: float | None = None,
    min_avg_duration: float | None = None,
    max_avg_duration: float | None = None,
    min_n_holdings: float | None = None,
    max_n_holdings: float | None = None,
) -> set:
    """Return the set of cik ids in `profile` matching every active filter.
    Each parameter set to None is skipped. Missing/NaN values are treated as
    failing the filter (never included by accident)."""
    if len(profile) == 0:
        return set()
    q = profile
    mask = pd.Series(True, index=q.index)
    if min_turnover is not None:
        mask &= q["turnover_mean"].fillna(-np.inf) >= min_turnover
    if max_turnover is not None:
        mask &= q["turnover_mean"].fillna(np.inf) <= max_turnover
    if max_open_rate is not None:
        mask &= q["open_rate_mean"].fillna(np.inf) <= max_open_rate
    if min_avg_duration is not None:
        mask &= q["avg_holding_duration"].fillna(-np.inf) >= min_avg_duration
    if max_avg_duration is not None:
        mask &= q["avg_holding_duration"].fillna(np.inf) <= max_avg_duration
    if min_n_holdings is not None:
        mask &= q["n_holdings_mean"].fillna(-np.inf) >= min_n_holdings
    if max_n_holdings is not None:
        mask &= q["n_holdings_mean"].fillna(np.inf) <= max_n_holdings
    if archetype is not None:
        if archetype not in _VALID_ARCHETYPES:
            raise ValueError(f"unknown archetype {archetype!r}; valid: "
                             f"{_VALID_ARCHETYPES}")
        col = f"is_{archetype}"
        if col not in q.columns:
            raise ValueError(f"profile missing column {col!r} — "
                             f"did you call tag_archetypes() first?")
        mask &= q[col].fillna(False).astype(bool)
    return set(q.loc[mask, "cik"].tolist())
