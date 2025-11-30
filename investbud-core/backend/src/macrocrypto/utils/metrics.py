"""
Portfolio performance metrics: Sharpe, Sortino, CAGR, Max Drawdown, VaR, CVaR, etc.
"""
import numpy as np
import pandas as pd
from typing import Optional


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns from prices."""
    return prices.pct_change().dropna()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from prices."""
    return np.log(prices / prices.shift(1)).dropna()


def calculate_cumulative_return(prices: pd.Series) -> float:
    """
    Calculate cumulative return.

    Args:
        prices: Series of prices

    Returns:
        Cumulative return (e.g., 0.25 = 25% gain)
    """
    return (prices.iloc[-1] / prices.iloc[0]) - 1


def calculate_cagr(prices: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        prices: Series of prices with datetime index

    Returns:
        CAGR (e.g., 0.15 = 15% annual return)
    """
    initial_value = prices.iloc[0]
    final_value = prices.iloc[-1]

    # Calculate time period in years
    days = (prices.index[-1] - prices.index[0]).days
    years = days / 365.25

    if years == 0:
        return 0.0

    cagr = (final_value / initial_value) ** (1 / years) - 1
    return cagr


def calculate_volatility(returns: pd.Series, annualize: bool = True) -> float:
    """
    Calculate volatility (standard deviation of returns).

    Args:
        returns: Series of returns
        annualize: If True, annualize the volatility

    Returns:
        Volatility
    """
    vol = returns.std()

    if annualize:
        # Annualize based on trading days (252 for stocks, 365 for crypto)
        vol = vol * np.sqrt(365)

    return vol


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 365
) -> float:
    """
    Calculate Sharpe Ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Trading periods per year (365 for crypto)

    Returns:
        Sharpe ratio
    """
    # Convert annual risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - rf_period

    if excess_returns.std() == 0:
        return 0.0

    # Annualize
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 365
) -> float:
    """
    Calculate Sortino Ratio (uses downside deviation instead of total volatility).

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Trading periods per year (365 for crypto)

    Returns:
        Sortino ratio
    """
    # Convert annual risk-free rate to period rate
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    excess_returns = returns - rf_period

    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std()

    # Annualize
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)

    return sortino


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        prices: Series of prices

    Returns:
        Maximum drawdown (e.g., -0.30 = -30% max drawdown)
    """
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    return drawdown.min()


def calculate_calmar_ratio(prices: pd.Series) -> float:
    """
    Calculate Calmar Ratio (CAGR / Max Drawdown).

    Args:
        prices: Series of prices

    Returns:
        Calmar ratio
    """
    cagr = calculate_cagr(prices)
    max_dd = abs(calculate_max_drawdown(prices))

    if max_dd == 0:
        return 0.0

    return cagr / max_dd


def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Series of returns
        confidence: Confidence level (default: 95%)

    Returns:
        VaR (e.g., -0.05 = 5% loss at 95% confidence)
    """
    return returns.quantile(1 - confidence)


def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    Args:
        returns: Series of returns
        confidence: Confidence level (default: 95%)

    Returns:
        CVaR (average loss beyond VaR threshold)
    """
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (percentage of positive return periods).

    Args:
        returns: Series of returns

    Returns:
        Win rate (e.g., 0.55 = 55% of periods had positive returns)
    """
    return (returns > 0).sum() / len(returns)


def calculate_all_metrics(
    prices: pd.Series,
    risk_free_rate: float = 0.02
) -> dict:
    """
    Calculate all portfolio metrics at once.

    Args:
        prices: Series of prices with datetime index
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of all metrics
    """
    returns = calculate_returns(prices)

    metrics = {
        'cumulative_return': calculate_cumulative_return(prices),
        'cagr': calculate_cagr(prices),
        'volatility': calculate_volatility(returns, annualize=True),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'max_drawdown': calculate_max_drawdown(prices),
        'calmar_ratio': calculate_calmar_ratio(prices),
        'var_95': calculate_var(returns, 0.95),
        'cvar_95': calculate_cvar(returns, 0.95),
        'win_rate': calculate_win_rate(returns),
        'total_periods': len(prices),
        'start_date': prices.index[0],
        'end_date': prices.index[-1],
        'start_value': prices.iloc[0],
        'end_value': prices.iloc[-1]
    }

    return metrics


def print_metrics_report(metrics: dict, strategy_name: str = "Strategy"):
    """
    Print a formatted metrics report.

    Args:
        metrics: Dictionary of metrics from calculate_all_metrics
        strategy_name: Name of the strategy
    """
    print("\n" + "=" * 70)
    print(f"{strategy_name.upper()} PERFORMANCE METRICS")
    print("=" * 70)

    print(f"\n--- Returns ---")
    print(f"Cumulative Return: {metrics['cumulative_return']*100:>10.2f}%")
    print(f"CAGR:              {metrics['cagr']*100:>10.2f}%")
    print(f"Start Value:       ${metrics['start_value']:>10,.2f}")
    print(f"End Value:         ${metrics['end_value']:>10,.2f}")

    print(f"\n--- Risk Metrics ---")
    print(f"Volatility (Ann.): {metrics['volatility']*100:>10.2f}%")
    print(f"Max Drawdown:      {metrics['max_drawdown']*100:>10.2f}%")
    print(f"VaR (95%):         {metrics['var_95']*100:>10.2f}%")
    print(f"CVaR (95%):        {metrics['cvar_95']*100:>10.2f}%")

    print(f"\n--- Risk-Adjusted Returns ---")
    print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>10.3f}")
    print(f"Sortino Ratio:     {metrics['sortino_ratio']:>10.3f}")
    print(f"Calmar Ratio:      {metrics['calmar_ratio']:>10.3f}")

    # Print advanced metrics if available
    if 'beta_vs_benchmark' in metrics:
        print(f"\n--- Advanced Metrics (vs Benchmark) ---")
        print(f"Beta:              {metrics.get('beta_vs_benchmark', 0):>10.3f}")
        print(f"Alpha (CAPM):      {metrics.get('capm_alpha', 0)*100:>10.2f}%")
        print(f"Excess Return:     {metrics.get('excess_return_vs_benchmark', 0)*100:>10.2f}%")
        print(f"Treynor Ratio:     {metrics.get('treynor_ratio', 0):>10.3f}")

    # Print fee information if available
    if 'total_fees' in metrics and metrics.get('num_swaps', 0) > 0:
        print(f"\n--- Trading Costs ---")
        print(f"Number of Swaps:   {metrics.get('num_swaps', 0):>10}")
        print(f"TX Fees:           ${metrics.get('total_tx_fees', 0):>10,.2f}")
        print(f"Slippage Loss:     ${metrics.get('total_slippage', 0):>10,.2f}")
        print(f"Pool Fees:         ${metrics.get('total_pool_fees', 0):>10,.2f}")
        print(f"Total Fees:        ${metrics.get('total_fees', 0):>10,.2f}")
        print(f"Avg per Swap:      ${metrics.get('avg_fee_per_swap', 0):>10,.2f}")

    print(f"\n--- Other ---")
    print(f"Win Rate:          {metrics['win_rate']*100:>10.2f}%")
    print(f"Total Periods:     {metrics['total_periods']:>10}")
    print(f"Date Range:        {metrics['start_date'].strftime('%Y-%m-%d')} to {metrics['end_date'].strftime('%Y-%m-%d')}")


def compare_metrics(metrics_dict: dict):
    """
    Compare metrics across multiple strategies.

    Args:
        metrics_dict: Dictionary of {strategy_name: metrics_dict}
    """
    comparison_df = pd.DataFrame(metrics_dict).T

    # Format columns
    pct_cols = ['cumulative_return', 'cagr', 'volatility', 'max_drawdown', 'var_95', 'cvar_95', 'win_rate',
                'excess_return_vs_benchmark', 'capm_alpha']
    for col in pct_cols:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col] * 100

    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)

    # Key metrics to display
    display_cols = [
        'cumulative_return', 'cagr', 'volatility', 'max_drawdown',
        'sharpe_ratio', 'sortino_ratio', 'win_rate'
    ]

    # Add advanced metrics if available
    advanced_cols = ['beta_vs_benchmark', 'capm_alpha', 'treynor_ratio', 'excess_return_vs_benchmark']
    for col in advanced_cols:
        if col in comparison_df.columns:
            display_cols.append(col)

    # Add fee metrics if available
    fee_cols = ['num_swaps', 'total_fees']
    for col in fee_cols:
        if col in comparison_df.columns:
            display_cols.append(col)

    display_df = comparison_df[[col for col in display_cols if col in comparison_df.columns]]

    print("\n" + display_df.to_string())

    # Highlight best performers
    print("\n" + "=" * 70)
    print("BEST PERFORMERS")
    print("=" * 70)

    metrics_to_highlight = {
        'Highest CAGR': ('cagr', True),
        'Best Sharpe': ('sharpe_ratio', True),
        'Best Sortino': ('sortino_ratio', True),
        'Lowest Volatility': ('volatility', False),
        'Smallest Drawdown': ('max_drawdown', False),
        'Highest Win Rate': ('win_rate', True),
        'Highest Alpha': ('capm_alpha', True),
        'Best Treynor Ratio': ('treynor_ratio', True)
    }

    for label, (metric, higher_is_better) in metrics_to_highlight.items():
        if metric in comparison_df.columns:
            if higher_is_better:
                best = comparison_df[metric].idxmax()
                value = comparison_df.loc[best, metric]
            else:
                best = comparison_df[metric].idxmin()
                value = comparison_df.loc[best, metric]

            print(f"{label:.<40} {best} ({value:.2f})")
