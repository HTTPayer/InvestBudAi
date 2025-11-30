"""
Grid search for optimal trading filters to reduce fee drag.

Tests three filtering mechanisms:
1. Confidence Threshold: Only trade when model is very confident
2. Minimum Holding Period: Minimum days between trades
3. Probability Change Threshold: Only trade when regime probability changes significantly
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from itertools import product

from src.macrocrypto.models import MacroRegimeClassifier, RegimeBacktester
from src.macrocrypto.data import CombinedDataPipeline


def apply_confidence_filter(
    backtest_df: pd.DataFrame,
    confidence_threshold: float = 0.75
) -> pd.Series:
    """
    Only trade when model confidence exceeds threshold.

    Args:
        backtest_df: DataFrame with predicted_regime and risk_on_probability
        confidence_threshold: Trade only if probability > threshold or < (1-threshold)

    Returns:
        Filtered regime series
    """
    filtered_regime = pd.Series(index=backtest_df.index, dtype=int)
    current_regime = 0  # Start in Risk-Off (cash)

    for i, (idx, row) in enumerate(backtest_df.iterrows()):
        prob = row['risk_on_probability']

        if prob > confidence_threshold:
            # High confidence Risk-On
            current_regime = 1
        elif prob < (1 - confidence_threshold):
            # High confidence Risk-Off
            current_regime = 0
        # else: maintain current regime

        filtered_regime.iloc[i] = current_regime

    return filtered_regime


def apply_holding_period_filter(
    backtest_df: pd.DataFrame,
    min_days: int = 7
) -> pd.Series:
    """
    Enforce minimum holding period between trades.

    Args:
        backtest_df: DataFrame with predicted_regime
        min_days: Minimum days to hold position before switching

    Returns:
        Filtered regime series
    """
    filtered_regime = pd.Series(index=backtest_df.index, dtype=int)
    current_regime = backtest_df['predicted_regime'].iloc[0]
    last_trade_date = backtest_df.index[0]

    for i, (idx, row) in enumerate(backtest_df.iterrows()):
        predicted = row['predicted_regime']
        days_since_trade = (idx - last_trade_date).days

        if i == 0:
            filtered_regime.iloc[i] = current_regime
        elif predicted != current_regime and days_since_trade >= min_days:
            # Switch regime only if enough time has passed
            current_regime = predicted
            last_trade_date = idx
            filtered_regime.iloc[i] = current_regime
        else:
            # Keep current regime
            filtered_regime.iloc[i] = current_regime

    return filtered_regime


def apply_probability_change_filter(
    backtest_df: pd.DataFrame,
    change_threshold: float = 0.3
) -> pd.Series:
    """
    Only trade when probability changes significantly.

    Args:
        backtest_df: DataFrame with risk_on_probability
        change_threshold: Trade only if probability changes by > this amount

    Returns:
        Filtered regime series
    """
    filtered_regime = pd.Series(index=backtest_df.index, dtype=int)
    current_regime = 0  # Start in Risk-Off
    last_prob = 0.0

    for i, (idx, row) in enumerate(backtest_df.iterrows()):
        prob = row['risk_on_probability']
        prob_change = abs(prob - last_prob)

        if i == 0:
            # Initial position based on first signal
            if prob > 0.5:
                current_regime = 1
                last_prob = prob
            filtered_regime.iloc[i] = current_regime
        elif prob_change > change_threshold:
            # Significant probability change - update regime and last_prob
            if prob > 0.5:
                current_regime = 1
            else:
                current_regime = 0
            last_prob = prob
            filtered_regime.iloc[i] = current_regime
        else:
            # Keep current regime, don't update last_prob
            filtered_regime.iloc[i] = current_regime

    return filtered_regime


def run_filtered_backtest(
    backtest_df: pd.DataFrame,
    filter_type: str,
    param_value: float,
    backtester: RegimeBacktester
) -> dict:
    """
    Run backtest with specified filter.

    Args:
        backtest_df: Prepared backtest data
        filter_type: 'confidence', 'holding_period', or 'prob_change'
        param_value: Parameter value for the filter
        backtester: RegimeBacktester instance

    Returns:
        Dictionary with results and metrics
    """
    # Apply filter
    if filter_type == 'confidence':
        filtered_regime = apply_confidence_filter(backtest_df, param_value)
    elif filter_type == 'holding_period':
        filtered_regime = apply_holding_period_filter(backtest_df, int(param_value))
    elif filter_type == 'prob_change':
        filtered_regime = apply_probability_change_filter(backtest_df, param_value)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Run simulation with filtered regime
    rf_rates = backtest_df['fed_funds']
    portfolio_values, fee_log = backtester.simulate_strategy(
        backtest_df['btc_price'],
        filtered_regime,
        strategy='model',
        risk_free_rate=rf_rates
    )

    # Calculate metrics
    from src.macrocrypto.utils.metrics import calculate_all_metrics

    metrics = calculate_all_metrics(portfolio_values, risk_free_rate=0.0437)

    # Add fee metrics
    total_fees = sum(f[0] for f in fee_log['total_fees'])
    num_swaps = len(fee_log['total_fees'])

    metrics['total_fees'] = total_fees
    metrics['num_swaps'] = num_swaps
    metrics['avg_fee_per_swap'] = total_fees / num_swaps if num_swaps > 0 else 0

    return {
        'filter_type': filter_type,
        'param_value': param_value,
        'portfolio_values': portfolio_values,
        'filtered_regime': filtered_regime,
        'fee_log': fee_log,
        'metrics': metrics
    }


def main():
    print("=" * 70)
    print("TRADING FILTER OPTIMIZATION")
    print("=" * 70)

    # Load model and data
    print("\n1. Loading model and data...")
    classifier = MacroRegimeClassifier()
    classifier.load('models/regime_classifier.pkl')

    pipeline = CombinedDataPipeline()
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    df = pipeline.fetch_combined_data(start_date=start_date)
    df = pipeline.create_risk_labels(df, method='btc_returns')

    # Prepare backtest data
    print("\n2. Preparing backtest data...")
    backtester = RegimeBacktester(initial_capital=10000)
    backtest_df = backtester.prepare_backtest_data(df, classifier)

    # Define parameter grids
    print("\n3. Setting up parameter grids...")
    grids = {
        'confidence': [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        'holding_period': [3, 5, 7, 10, 14, 21, 30],
        'prob_change': [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    }

    # Run grid search
    results = []

    for filter_type, param_grid in grids.items():
        print(f"\n{'='*70}")
        print(f"Testing {filter_type.upper()} filter...")
        print(f"{'='*70}")

        for param_value in param_grid:
            result = run_filtered_backtest(backtest_df, filter_type, param_value, backtester)
            results.append(result)

            m = result['metrics']
            print(f"\n{filter_type}={param_value:.2f}: "
                  f"CAGR={m['cagr']*100:>6.2f}% | "
                  f"Sharpe={m['sharpe_ratio']:>5.3f} | "
                  f"Swaps={m['num_swaps']:>3} | "
                  f"Fees=${m['total_fees']:>7,.0f} | "
                  f"Final=${m['end_value']:>10,.2f}")

    # Find best strategies
    print("\n" + "=" * 70)
    print("BEST STRATEGIES BY METRIC")
    print("=" * 70)

    # Best CAGR
    best_cagr = max(results, key=lambda x: x['metrics']['cagr'])
    print(f"\nHighest CAGR: {best_cagr['filter_type']}={best_cagr['param_value']:.2f}")
    print(f"  CAGR: {best_cagr['metrics']['cagr']*100:.2f}%")
    print(f"  Sharpe: {best_cagr['metrics']['sharpe_ratio']:.3f}")
    print(f"  Swaps: {best_cagr['metrics']['num_swaps']}")
    print(f"  Total Fees: ${best_cagr['metrics']['total_fees']:,.2f}")
    print(f"  Final Value: ${best_cagr['metrics']['end_value']:,.2f}")

    # Best Sharpe
    best_sharpe = max(results, key=lambda x: x['metrics']['sharpe_ratio'])
    print(f"\nBest Sharpe Ratio: {best_sharpe['filter_type']}={best_sharpe['param_value']:.2f}")
    print(f"  CAGR: {best_sharpe['metrics']['cagr']*100:.2f}%")
    print(f"  Sharpe: {best_sharpe['metrics']['sharpe_ratio']:.3f}")
    print(f"  Swaps: {best_sharpe['metrics']['num_swaps']}")
    print(f"  Total Fees: ${best_sharpe['metrics']['total_fees']:,.2f}")
    print(f"  Final Value: ${best_sharpe['metrics']['end_value']:,.2f}")

    # Fewest trades
    best_trades = min(results, key=lambda x: x['metrics']['num_swaps'])
    print(f"\nFewest Trades: {best_trades['filter_type']}={best_trades['param_value']:.2f}")
    print(f"  CAGR: {best_trades['metrics']['cagr']*100:.2f}%")
    print(f"  Sharpe: {best_trades['metrics']['sharpe_ratio']:.3f}")
    print(f"  Swaps: {best_trades['metrics']['num_swaps']}")
    print(f"  Total Fees: ${best_trades['metrics']['total_fees']:,.2f}")
    print(f"  Final Value: ${best_trades['metrics']['end_value']:,.2f}")

    # Create comparison DataFrame
    print("\n" + "=" * 70)
    print("ALL RESULTS COMPARISON")
    print("=" * 70)

    comparison_data = []
    for r in results:
        m = r['metrics']
        comparison_data.append({
            'filter': r['filter_type'],
            'param': r['param_value'],
            'cagr': m['cagr'] * 100,
            'sharpe': m['sharpe_ratio'],
            'swaps': m['num_swaps'],
            'total_fees': m['total_fees'],
            'final_value': m['end_value']
        })

    comp_df = pd.DataFrame(comparison_data)

    # Show top 10 by CAGR
    print("\nTop 10 by CAGR:")
    print(comp_df.nlargest(10, 'cagr').to_string(index=False))

    # Show top 10 by Sharpe
    print("\nTop 10 by Sharpe Ratio:")
    print(comp_df.nlargest(10, 'sharpe').to_string(index=False))

    # Add baseline comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)

    from src.macrocrypto.utils.metrics import calculate_all_metrics as calc_metrics

    # Run buy & hold
    buy_hold_values, buy_hold_fees = backtester.simulate_strategy(
        backtest_df['btc_price'],
        pd.Series(1, index=backtest_df.index),
        strategy='buy_and_hold',
        risk_free_rate=backtest_df['fed_funds']
    )
    buy_hold_metrics = calc_metrics(buy_hold_values, 0.0437)

    # Run unfiltered model
    unfiltered_values, unfiltered_fees = backtester.simulate_strategy(
        backtest_df['btc_price'],
        backtest_df['predicted_regime'],
        strategy='model',
        risk_free_rate=backtest_df['fed_funds']
    )
    unfiltered_metrics = calc_metrics(unfiltered_values, 0.0437)
    unfiltered_total_fees = sum(f[0] for f in unfiltered_fees['total_fees'])
    unfiltered_num_swaps = len(unfiltered_fees['total_fees'])

    print(f"\nBuy & Hold:")
    print(f"  CAGR: {buy_hold_metrics['cagr']*100:.2f}%")
    print(f"  Sharpe: {buy_hold_metrics['sharpe_ratio']:.3f}")
    print(f"  Swaps: 1")
    print(f"  Total Fees: $135.00")
    print(f"  Final Value: ${buy_hold_metrics['end_value']:,.2f}")

    print(f"\nUnfiltered Model:")
    print(f"  CAGR: {unfiltered_metrics['cagr']*100:.2f}%")
    print(f"  Sharpe: {unfiltered_metrics['sharpe_ratio']:.3f}")
    print(f"  Swaps: {unfiltered_num_swaps}")
    print(f"  Total Fees: ${unfiltered_total_fees:,.2f}")
    print(f"  Final Value: ${unfiltered_metrics['end_value']:,.2f}")

    print(f"\nBest Filtered Strategy ({best_cagr['filter_type']}={best_cagr['param_value']:.2f}):")
    print(f"  CAGR: {best_cagr['metrics']['cagr']*100:.2f}% ({best_cagr['metrics']['cagr']*100 - buy_hold_metrics['cagr']*100:+.2f}% vs B&H)")
    print(f"  Sharpe: {best_cagr['metrics']['sharpe_ratio']:.3f} ({best_cagr['metrics']['sharpe_ratio'] - buy_hold_metrics['sharpe_ratio']:+.3f} vs B&H)")
    print(f"  Swaps: {best_cagr['metrics']['num_swaps']} ({best_cagr['metrics']['num_swaps'] - unfiltered_num_swaps:+d} vs Unfiltered)")
    print(f"  Total Fees: ${best_cagr['metrics']['total_fees']:,.2f} (${best_cagr['metrics']['total_fees'] - unfiltered_total_fees:+,.2f} vs Unfiltered)")
    print(f"  Final Value: ${best_cagr['metrics']['end_value']:,.2f}")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
