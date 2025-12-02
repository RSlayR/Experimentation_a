"""
CFA-inspired portfolio optimizer with risk-profile-aware allocation and
ridge/lasso regularization to control concentration.

The script downloads ETF/stock data from Yahoo Finance, computes CAPM-style
expected returns, and optimizes a mean-variance objective with optional
penalties:
- L1 (lasso) to reduce the number of tickers.
- L2 (ridge) to reduce weight magnitude.
- A stock penalty to prefer ETFs when both are available.

The script also plots the optimized portfolio relative to the risk-free rate
and a chosen market proxy, annotating common performance statistics such as
Sharpe, Treynor, Jensen's alpha, and Modigliani-Modigliani (M²).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import yfinance as yf

TRADING_DAYS = 252


@dataclasses.dataclass
class RiskProfile:
    """Parameter bundle describing a risk appetite and constraints."""

    name: str
    risk_aversion: float
    l1_penalty: float
    l2_penalty: float
    max_single_allocation: float
    horizon_years: int


DEFAULT_RISK_PROFILES: Mapping[str, RiskProfile] = {
    "conservative": RiskProfile(
        name="Conservative",
        risk_aversion=6.0,
        l1_penalty=0.08,
        l2_penalty=0.05,
        max_single_allocation=0.25,
        horizon_years=3,
    ),
    "balanced": RiskProfile(
        name="Balanced",
        risk_aversion=3.5,
        l1_penalty=0.05,
        l2_penalty=0.04,
        max_single_allocation=0.3,
        horizon_years=7,
    ),
    "growth": RiskProfile(
        name="Growth",
        risk_aversion=2.0,
        l1_penalty=0.02,
        l2_penalty=0.03,
        max_single_allocation=0.35,
        horizon_years=12,
    ),
}


def fetch_price_history(
    tickers: Iterable[str], start: str | dt.date, end: str | dt.date
) -> pd.DataFrame:
    """Download adjusted close prices for the requested tickers."""

    data = yf.download(list(tickers), start=start, end=end, progress=False)["Adj Close"]
    return data.dropna(how="all")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""

    return np.log(prices / prices.shift(1)).dropna()


def annualize_returns(returns: pd.Series) -> pd.Series:
    return returns.mean() * TRADING_DAYS


def annualize_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS


def compute_beta(
    asset_returns: pd.DataFrame, market_returns: pd.Series
) -> pd.Series:
    """Estimate CAPM beta for each asset against the market series."""

    joined = asset_returns.join(market_returns.rename("market"), how="inner")
    betas = {}
    for col in asset_returns.columns:
        cov = np.cov(joined[col], joined["market"])[0, 1]
        market_var = np.var(joined["market"])
        betas[col] = cov / market_var if market_var != 0 else 0.0
    return pd.Series(betas)


def capm_expected_returns(
    risk_free_rate: float, betas: pd.Series, market_return: float
) -> pd.Series:
    """Compute expected returns via CAPM."""

    return risk_free_rate + betas * (market_return - risk_free_rate)


def optimize_portfolio(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_profile: RiskProfile,
    asset_types: Mapping[str, str],
    etf_preference_penalty: float = 0.05,
) -> Tuple[np.ndarray, float, float]:
    """Optimize a regularized mean-variance portfolio.

    Objective: minimize
        risk_aversion * w^T C w - mu^T w
        + etf_penalty * sum(w for non-ETF assets)
        + l1 * ||w||_1 + l2 * ||w||_2^2
    Subject to: sum(w) = 1, 0 <= w_i <= max_single_allocation
    """

    mu = expected_returns.values
    cov_matrix = cov.values
    asset_list = list(expected_returns.index)
    is_stock = np.array([1.0 if asset_types.get(t, "stock").lower() != "etf" else 0.0 for t in asset_list])

    def objective(weights: np.ndarray) -> float:
        variance = weights.T @ cov_matrix @ weights
        ret = mu @ weights
        stock_penalty = etf_preference_penalty * float(is_stock @ weights)
        l1 = risk_profile.l1_penalty * np.sum(np.abs(weights))
        l2 = risk_profile.l2_penalty * np.sum(weights**2)
        return risk_profile.risk_aversion * variance - ret + stock_penalty + l1 + l2

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, risk_profile.max_single_allocation)] * len(asset_list)
    initial_w = np.full(len(asset_list), 1 / len(asset_list))

    result = opt.minimize(objective, initial_w, bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = result.x
    port_return = mu @ weights
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return weights, port_return, port_vol


def portfolio_statistics(
    weights: np.ndarray,
    returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float,
) -> Dict[str, float]:
    """Compute Sharpe, Treynor, Jensen alpha, and M-squared."""

    portfolio_returns = returns @ weights
    ann_return = annualize_returns(portfolio_returns)
    ann_vol = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
    beta = np.cov(portfolio_returns, market_returns)[0, 1] / np.var(market_returns)

    sharpe = (ann_return - risk_free_rate) / ann_vol
    treynor = (ann_return - risk_free_rate) / beta

    market_ann_return = annualize_returns(market_returns)
    alpha = ann_return - (risk_free_rate + beta * (market_ann_return - risk_free_rate))
    m2 = risk_free_rate + sharpe * (market_returns.std() * np.sqrt(TRADING_DAYS))

    return {
        "annual_return": ann_return,
        "annual_vol": ann_vol,
        "beta": beta,
        "sharpe": sharpe,
        "treynor": treynor,
        "jensen_alpha": alpha,
        "m2": m2,
    }


def plot_positions(
    portfolio_point: Tuple[float, float],
    market_point: Tuple[float, float],
    risk_free_rate: float,
    stats: Mapping[str, float],
    title: str,
) -> None:
    """Plot the portfolio vs risk-free and market portfolios."""

    port_vol, port_ret = portfolio_point
    market_vol, market_ret = market_point

    plt.figure(figsize=(10, 6))
    plt.scatter([0], [risk_free_rate], color="green", label="Risk-free")
    plt.scatter([market_vol], [market_ret], color="blue", label="Market portfolio")
    plt.scatter([port_vol], [port_ret], color="orange", label="Optimized portfolio")

    # Draw capital market line
    cml_x = np.linspace(0, max(port_vol, market_vol) * 1.2, 50)
    market_sharpe = (market_ret - risk_free_rate) / market_vol
    cml_y = risk_free_rate + market_sharpe * cml_x
    plt.plot(cml_x, cml_y, linestyle="--", color="gray", label="Capital Market Line")

    plt.title(title)
    plt.xlabel("Volatility (σ)")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    text = (
        f"Sharpe: {stats['sharpe']:.2f}\n"
        f"Treynor: {stats['treynor']:.2f}\n"
        f"Jensen Alpha: {stats['jensen_alpha']:.2%}\n"
        f"M²: {stats['m2']:.2%}"
    )
    plt.annotate(text, xy=(port_vol, port_ret), xytext=(port_vol * 1.05, port_ret * 0.9),
                 bbox=dict(boxstyle="round", fc="w", ec="0.8"))
    plt.tight_layout()


def demo():
    """Run an end-to-end optimization example with default settings."""

    start = (dt.date.today() - dt.timedelta(days=365 * 3))
    end = dt.date.today()

    etfs = ["VTI", "VEA", "AGG", "VNQ"]
    stocks = ["AAPL", "MSFT", "GOOGL"]
    market_proxy = "SPY"

    tickers = etfs + stocks + [market_proxy]
    prices = fetch_price_history(tickers, start, end)
    returns = compute_returns(prices)

    market_returns = returns[market_proxy]
    asset_returns = returns.drop(columns=[market_proxy])

    betas = compute_beta(asset_returns, market_returns)
    market_ann_return = annualize_returns(market_returns)

    risk_free_rate = 0.02
    profile = DEFAULT_RISK_PROFILES["balanced"]
    expected = capm_expected_returns(risk_free_rate, betas, market_ann_return)
    cov = annualize_covariance(asset_returns)

    asset_types = {t: "etf" for t in etfs}
    asset_types.update({t: "stock" for t in stocks})

    weights, port_ret, port_vol = optimize_portfolio(
        expected_returns=expected,
        cov=cov,
        risk_profile=profile,
        asset_types=asset_types,
        etf_preference_penalty=0.07,
    )

    stats = portfolio_statistics(weights, asset_returns, market_returns, risk_free_rate)
    plot_positions(
        portfolio_point=(port_vol, port_ret),
        market_point=(market_returns.std() * np.sqrt(TRADING_DAYS), market_ann_return),
        risk_free_rate=risk_free_rate,
        stats=stats,
        title=f"{profile.name} portfolio vs. market",
    )

    allocation = pd.Series(weights, index=asset_returns.columns)
    print("Optimized weights:\n", allocation.sort_values(ascending=False))
    print("\nPerformance stats:")
    for k, v in stats.items():
        print(f"{k:15s}: {v:.4f}")

    plt.show()


if __name__ == "__main__":
    demo()
