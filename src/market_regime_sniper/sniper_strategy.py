from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_regime_ids(df_clusters: pd.DataFrame, cluster_col: str = "cluster", value_col: str = "returns"):
    """
    Identify bear / aggressive bull / Sharpe-optimal bull regimes based on annual returns.
    """
    stats = df_clusters.groupby(cluster_col)[value_col].agg(["mean", "std", "count"])
    stats["sharpe"] = stats["mean"] / (stats["std"] + 1e-9)

    bear_id = stats["mean"].idxmin()
    agg_bull_id = stats["mean"].idxmax()
    smart_bull_id = stats["sharpe"].idxmax()

    special_ids = {bear_id, agg_bull_id, smart_bull_id}
    other_ids = [c for c in stats.index if c not in special_ids]

    return {
        "bear": bear_id,
        "agg_bull": agg_bull_id,
        "smart_bull": smart_bull_id,
        "others": other_ids,
        "stats": stats.sort_values(by="mean", ascending=False),
    }


def get_metrics(equity_curve: pd.Series, invested_curve: pd.Series) -> dict:
    """
    Compute core performance metrics: CAGR, max drawdown, Sharpe, Calmar.
    """
    returns = equity_curve.pct_change().dropna()
    final = float(equity_curve.iloc[-1])
    inv = float(invested_curve.iloc[-1])

    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    cagr = (final / inv) ** (365.25 / days) - 1.0 if days > 0 else 0.0

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    rf = 0.03 / 252
    exc = returns - rf
    sharpe = (exc.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    return {
        "Final Equity ($)": round(final, 2),
        "CAGR (%)": round(cagr * 100, 2),
        "Max DD (%)": round(max_dd * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Calmar": round(calmar, 3),
    }


def run_hybrid_sniper(
    price_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    daily_budget: float = 100.0,
    bear_exposure: float = 0.5,
    bull_exposure_base: float = 2.5,
    ma_window: int = 200,
) -> pd.DataFrame:
    """
    Hybrid Sniper strategy with regime-aware exposure and simple volatility control.

    - Uses lagged annual regime labels (year t uses label of year t-1)
    - Bear regime: reduce exposure (cash defensive stance)
    - Bull regime: trend-following with MA filter and volatility scaling
    - Neutral regimes: classic DCA
    """
    reg_info = get_regime_ids(clusters_df, cluster_col="cluster", value_col="returns")
    bear_cluster = reg_info["bear"]
    bull_cluster = reg_info["agg_bull"]

    df = price_df[["Close"]].copy()
    df["Year"] = df.index.year

    cluster_map = clusters_df["cluster"].to_dict()
    df["Regime"] = df["Year"].apply(lambda y: cluster_map.get(y - 1, np.nan))

    df["MA"] = df["Close"].rolling(window=ma_window).mean()
    df["Vol"] = df["Close"].pct_change().rolling(63).std() * np.sqrt(252)

    stats: list[dict] = []

    df["Eq_Base"] = (daily_budget / df["Close"]).cumsum() * df["Close"]
    inv = pd.Series(np.arange(1, len(df) + 1) * daily_budget, index=df.index)
    m_base = get_metrics(df["Eq_Base"], inv)
    stats.append({"Strategy": "Baseline (DCA)", **m_base})

    equity: list[float] = []
    invested: list[float] = []
    curr_eq = 0.0
    total_inv = 0.0

    CASH_YIELD = 0.03 / 252
    LEV_COST = 0.05 / 252
    vol_target_annual = 0.18

    prices = df["Close"].values
    mas = df["MA"].values
    regimes = df["Regime"].values
    vols = df["Vol"].values

    for i in range(len(df)):
        curr_eq += daily_budget
        total_inv += daily_budget

        mkt_ret = (prices[i] / prices[i - 1]) - 1.0 if i > 0 else 0.0

        raw_cluster = regimes[i]
        price = prices[i]
        ma = mas[i]
        current_vol = vols[i]

        exposure = 1.0

        if not np.isnan(raw_cluster):
            c_id = int(raw_cluster)

            if c_id == bear_cluster:
                exposure = bear_exposure
            elif c_id == bull_cluster:
                if not np.isnan(ma) and price > ma:
                    if not np.isnan(current_vol) and current_vol > 0:
                        vol_scalar = vol_target_annual / current_vol
                        exposure = float(np.clip(vol_scalar * bull_exposure_base, 0.5, bull_exposure_base))
                    else:
                        exposure = 1.0
                else:
                    exposure = 1.0
            else:
                exposure = 1.0

        if exposure <= 1.0:
            port_ret = (mkt_ret * exposure) + (CASH_YIELD * (1 - exposure))
        else:
            port_ret = (mkt_ret * exposure) - (LEV_COST * (exposure - 1))

        curr_eq = curr_eq * (1 + port_ret)
        equity.append(curr_eq)
        invested.append(total_inv)

    s_eq = pd.Series(equity, index=df.index)
    s_inv = pd.Series(invested, index=df.index)
    m_strat = get_metrics(s_eq, s_inv)
    stats.append({"Strategy": "Hybrid Sniper (Vol Scaled)", **m_strat})

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 14), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )

    ax1.plot(df.index, df["Eq_Base"], "k--", alpha=0.5, label="Baseline (DCA)")
    ax1.plot(df.index, s_eq, "#2ca02c", linewidth=2, label="Hybrid Sniper")
    ax1.set_yscale("log")
    alpha_amount = m_strat["Final Equity ($)"] - m_base["Final Equity ($)"]
    ax1.set_title(
        f"Hybrid Sniper Strategy\nTotal Alpha Generated: +${alpha_amount:,.0f}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Value ($, Log)")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, which="both", alpha=0.2)

    dd_base = (df["Eq_Base"] - df["Eq_Base"].cummax()) / df["Eq_Base"].cummax()
    dd_strat = (s_eq - s_eq.cummax()) / s_eq.cummax()
    ax2.plot(df.index, dd_base * 100, "k--", alpha=0.4, label="Baseline DD")
    ax2.plot(df.index, dd_strat * 100, "#d62728", linewidth=1, label="Sniper DD")
    ax2.fill_between(df.index, 0, -100, color="red", alpha=0.05)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Risk Profile: Drawdown Comparison")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    alpha_curve = s_eq - df["Eq_Base"]
    ax3.fill_between(
        df.index,
        alpha_curve,
        0,
        where=(alpha_curve >= 0),
        color="#2ca02c",
        alpha=0.3,
        label="Outperformance",
    )
    ax3.fill_between(
        df.index,
        alpha_curve,
        0,
        where=(alpha_curve < 0),
        color="#d62728",
        alpha=0.3,
        label="Underperformance",
    )
    ax3.plot(df.index, alpha_curve, "k-", linewidth=0.8, alpha=0.6)
    ax3.set_ylabel("Net Alpha ($)")
    ax3.set_title("Cumulative Cash Advantage vs Baseline")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(stats).set_index("Strategy")
