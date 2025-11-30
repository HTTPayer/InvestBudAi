"""
Advanced portfolio metrics including beta, excess return, Treynor ratio, and CAPM alpha.
Based on portfolio_optimizers/classifier_optimizer/python_scripts/utils.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Optional


def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate portfolio beta relative to benchmark.

    Beta measures the volatility of the portfolio relative to the benchmark.
    - Beta > 1: More volatile than benchmark
    - Beta = 1: Same volatility as benchmark
    - Beta < 1: Less volatile than benchmark

    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series (e.g., BTC)

    Returns:
        Beta coefficient
    """
    # Align indices
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)

    if len(common_index) < 2:
        return 0.0

    portfolio_returns = portfolio_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]

    # Calculate percentage returns if not already
    if not all(portfolio_returns.between(-1, 10)):  # Assumes returns are not already pct
        portfolio_returns = portfolio_returns.pct_change().dropna()
        benchmark_returns = benchmark_returns.pct_change().dropna()

    # Remove NaN and infinite values
    portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan).dropna()
    benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan).dropna()

    # Align again after dropna
    common_index = portfolio_returns.index.intersection(benchmark_returns.index)
    portfolio_returns = portfolio_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]

    if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0

    # Reshape for sklearn
    X = benchmark_returns.values.reshape(-1, 1)
    Y = portfolio_returns.values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, Y)

    beta = model.coef_[0]
    return float(beta)


def calculate_excess_return(
    portfolio_return: float,
    benchmark_return: float
) -> float:
    """
    Calculate excess return of portfolio over benchmark.

    Args:
        portfolio_return: Total portfolio return (e.g., 0.15 for 15%)
        benchmark_return: Total benchmark return (e.g., 0.10 for 10%)

    Returns:
        Excess return (alpha)
    """
    return portfolio_return - benchmark_return


def calculate_treynor_ratio(
    portfolio_return: float,
    risk_free_rate: float,
    beta: float
) -> float:
    """
    Calculate Treynor ratio (risk-adjusted return using beta).

    Treynor ratio = (Portfolio Return - Risk-Free Rate) / Beta

    Measures return earned in excess of risk-free rate per unit of market risk (beta).
    Higher is better.

    Args:
        portfolio_return: Annualized portfolio return
        risk_free_rate: Annualized risk-free rate
        beta: Portfolio beta

    Returns:
        Treynor ratio
    """
    if beta == 0:
        return 0.0

    excess_return = portfolio_return - risk_free_rate
    treynor = excess_return / beta

    return float(treynor)


def calculate_alpha(
    portfolio_return: float,
    benchmark_return: float,
    risk_free_rate: float,
    beta: float
) -> float:
    """
    Calculate CAPM alpha (Jensen's alpha).

    Alpha = Portfolio Return - [Risk-Free Rate + Beta * (Benchmark Return - Risk-Free Rate)]

    Measures excess return above what CAPM predicts.
    - Positive alpha: Outperformed expectations
    - Negative alpha: Underperformed expectations

    Args:
        portfolio_return: Annualized portfolio return
        benchmark_return: Annualized benchmark return
        risk_free_rate: Annualized risk-free rate
        beta: Portfolio beta

    Returns:
        Alpha (excess return above CAPM prediction)
    """
    # Expected return according to CAPM
    expected_return = risk_free_rate + beta * (benchmark_return - risk_free_rate)

    # Alpha is actual return minus expected return
    alpha = portfolio_return - expected_return

    return float(alpha)


def calculate_all_advanced_metrics(
    portfolio_prices: pd.Series,
    benchmark_prices: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate all advanced portfolio metrics.

    Args:
        portfolio_prices: Portfolio value time series
        benchmark_prices: Benchmark price time series (e.g., BTC prices)
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with all advanced metrics
    """
    from .metrics import calculate_cagr

    # Calculate returns
    portfolio_returns = portfolio_prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()

    # Calculate CAGRs
    portfolio_cagr = calculate_cagr(portfolio_prices)
    benchmark_cagr = calculate_cagr(benchmark_prices)

    # Calculate beta
    beta = calculate_beta(portfolio_returns, benchmark_returns)

    # Calculate excess return
    excess_return = calculate_excess_return(portfolio_cagr, benchmark_cagr)

    # Calculate Treynor ratio
    treynor = calculate_treynor_ratio(portfolio_cagr, risk_free_rate, beta)

    # Calculate alpha
    alpha = calculate_alpha(portfolio_cagr, benchmark_cagr, risk_free_rate, beta)

    return {
        'beta_vs_benchmark': round(beta, 4),
        'excess_return_vs_benchmark': round(excess_return, 4),
        'treynor_ratio': round(treynor, 4),
        'capm_alpha': round(alpha, 4),
        'portfolio_cagr': round(portfolio_cagr, 4),
        'benchmark_cagr': round(benchmark_cagr, 4)
    }


def calculate_portfolio_vs_btc_metrics(
    portfolio_prices: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate portfolio metrics vs BTC benchmark.

    This is a convenience function that fetches BTC data and calculates
    all advanced metrics against it.

    Args:
        portfolio_prices: Portfolio value time series with datetime index
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with advanced metrics vs BTC
    """
    import yfinance as yf

    # Get BTC data for the same period
    start_date = portfolio_prices.index[0]
    end_date = portfolio_prices.index[-1]

    # Fetch BTC data
    btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)

    if btc_data.empty:
        print("[WARNING] Could not fetch BTC data for comparison")
        return {
            'beta_vs_btc': None,
            'excess_return_vs_btc': None,
            'treynor_ratio': None,
            'capm_alpha': None
        }

    # Use closing prices
    btc_prices = btc_data['Close']

    # Align indices (portfolio might be hourly, BTC is daily)
    # Resample portfolio to daily if needed
    if len(portfolio_prices) > len(btc_prices) * 2:  # Likely hourly data
        portfolio_prices_daily = portfolio_prices.resample('D').last().dropna()
    else:
        portfolio_prices_daily = portfolio_prices

    # Align dates
    common_dates = portfolio_prices_daily.index.intersection(btc_prices.index)
    portfolio_prices_aligned = portfolio_prices_daily.loc[common_dates]
    btc_prices_aligned = btc_prices.loc[common_dates]

    if len(common_dates) < 2:
        print("[WARNING] Insufficient overlapping data for BTC comparison")
        return {
            'beta_vs_btc': None,
            'excess_return_vs_btc': None,
            'treynor_ratio': None,
            'capm_alpha': None
        }

    # Calculate metrics
    return calculate_all_advanced_metrics(
        portfolio_prices_aligned,
        btc_prices_aligned,
        risk_free_rate
    )
