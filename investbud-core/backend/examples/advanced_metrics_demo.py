"""
Demonstration of advanced portfolio metrics.

This shows how to calculate beta, excess return, Treynor ratio, and CAPM alpha
when you have historical portfolio values.

Run with: uv run python examples/advanced_metrics_demo.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.macrocrypto.utils.advanced_metrics import (
    calculate_beta,
    calculate_excess_return,
    calculate_treynor_ratio,
    calculate_alpha,
    calculate_all_advanced_metrics,
    calculate_portfolio_vs_btc_metrics
)
from src.macrocrypto.utils.metrics import calculate_all_metrics


def generate_sample_portfolio_data():
    """Generate sample portfolio and benchmark data for demonstration."""
    # Generate 365 days of data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

    # Simulate portfolio values (starting at $10,000)
    np.random.seed(42)
    portfolio_returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily mean, 2% std
    portfolio_values = [10000]

    for ret in portfolio_returns[1:]:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))

    portfolio_series = pd.Series(portfolio_values, index=dates, name='Portfolio')

    # Simulate benchmark (BTC) values (starting at $40,000)
    benchmark_returns = np.random.normal(0.0003, 0.03, len(dates))  # 0.03% daily mean, 3% std
    benchmark_values = [40000]

    for ret in benchmark_returns[1:]:
        benchmark_values.append(benchmark_values[-1] * (1 + ret))

    benchmark_series = pd.Series(benchmark_values, index=dates, name='BTC')

    return portfolio_series, benchmark_series


def main():
    print("=" * 70)
    print("Advanced Portfolio Metrics Demonstration")
    print("=" * 70)

    # Generate sample data
    print("\n1. Generating sample portfolio data...")
    portfolio, benchmark = generate_sample_portfolio_data()

    print(f"\nPortfolio:")
    print(f"  Start Value: ${portfolio.iloc[0]:,.2f}")
    print(f"  End Value: ${portfolio.iloc[-1]:,.2f}")
    print(f"  Total Return: {(portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100:.2f}%")

    print(f"\nBenchmark (BTC):")
    print(f"  Start Value: ${benchmark.iloc[0]:,.2f}")
    print(f"  End Value: ${benchmark.iloc[-1]:,.2f}")
    print(f"  Total Return: {(benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100:.2f}%")

    # Calculate basic metrics
    print("\n" + "=" * 70)
    print("2. Basic Portfolio Metrics")
    print("=" * 70)

    risk_free_rate = 0.02
    basic_metrics = calculate_all_metrics(portfolio, risk_free_rate)

    print(f"\nCAGR: {basic_metrics['cagr'] * 100:.2f}%")
    print(f"Sharpe Ratio: {basic_metrics['sharpe']:.3f}")
    print(f"Sortino Ratio: {basic_metrics['sortino']:.3f}")
    print(f"Max Drawdown: {basic_metrics['max_drawdown'] * 100:.2f}%")
    print(f"Volatility: {basic_metrics['volatility'] * 100:.2f}%")

    # Calculate advanced metrics
    print("\n" + "=" * 70)
    print("3. Advanced Metrics (vs Benchmark)")
    print("=" * 70)

    advanced_metrics = calculate_all_advanced_metrics(portfolio, benchmark, risk_free_rate)

    print(f"\nBeta (vs Benchmark): {advanced_metrics['beta_vs_benchmark']:.3f}")
    print(f"  → Portfolio is {abs(advanced_metrics['beta_vs_benchmark'] - 1) * 100:.1f}% ", end="")
    if advanced_metrics['beta_vs_benchmark'] > 1:
        print(f"MORE volatile than benchmark")
    else:
        print(f"LESS volatile than benchmark")

    print(f"\nExcess Return (vs Benchmark): {advanced_metrics['excess_return_vs_benchmark'] * 100:.2f}%")
    if advanced_metrics['excess_return_vs_benchmark'] > 0:
        print(f"  → Portfolio OUTPERFORMED benchmark")
    else:
        print(f"  → Portfolio UNDERPERFORMED benchmark")

    print(f"\nTreynor Ratio: {advanced_metrics['treynor_ratio']:.3f}")
    print(f"  → Risk-adjusted return per unit of market risk (beta)")

    print(f"\nCAPM Alpha: {advanced_metrics['capm_alpha'] * 100:.2f}%")
    if advanced_metrics['capm_alpha'] > 0:
        print(f"  → Portfolio generated POSITIVE alpha (outperformed CAPM expectations)")
    else:
        print(f"  → Portfolio generated NEGATIVE alpha (underperformed CAPM expectations)")

    # Individual metric calculations
    print("\n" + "=" * 70)
    print("4. Individual Metric Calculations")
    print("=" * 70)

    beta = calculate_beta(portfolio.pct_change().dropna(), benchmark.pct_change().dropna())
    print(f"\nBeta: {beta:.3f}")

    portfolio_return = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (365 / len(portfolio)) - 1
    benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0]) ** (365 / len(benchmark)) - 1

    excess = calculate_excess_return(portfolio_return, benchmark_return)
    print(f"Excess Return: {excess * 100:.2f}%")

    treynor = calculate_treynor_ratio(portfolio_return, risk_free_rate, beta)
    print(f"Treynor Ratio: {treynor:.3f}")

    alpha = calculate_alpha(portfolio_return, benchmark_return, risk_free_rate, beta)
    print(f"CAPM Alpha: {alpha * 100:.2f}%")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("• Beta measures portfolio volatility vs benchmark")
    print("• Excess Return shows outperformance/underperformance")
    print("• Treynor Ratio adjusts returns for systematic risk (beta)")
    print("• CAPM Alpha shows risk-adjusted outperformance")
    print("\nTo use with real data:")
    print("  metrics = calculate_all_advanced_metrics(portfolio_prices, btc_prices)")


if __name__ == '__main__':
    main()
