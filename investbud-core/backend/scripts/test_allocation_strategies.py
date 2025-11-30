"""
Test different allocation strategies to minimize fees and maximize returns.

Strategies tested:
1. Binary (100/0): Current approach - all-in or all-out
2. Gradual (70/30, 75/25, 80/20): Fixed allocations for Risk-On/Risk-Off
3. Dynamic: Use model probability directly as allocation
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

from src.macrocrypto.models import MacroRegimeClassifier
from src.macrocrypto.data import CombinedDataPipeline
from src.macrocrypto.utils.metrics import calculate_all_metrics


class AllocationBacktester:
    """Backtester with support for various allocation strategies."""

    def __init__(
        self,
        initial_capital: float = 10000,
        slippage_decimal: float = 0.01,
        pool_fee_decimal: float = 0.003,
        tx_fee_usd: float = 5.0,
        min_holding_days: int = 21
    ):
        self.initial_capital = initial_capital
        self.slippage_decimal = slippage_decimal
        self.pool_fee_decimal = pool_fee_decimal
        self.tx_fee_usd = tx_fee_usd
        self.min_holding_days = min_holding_days

    def simulate_allocation_strategy(
        self,
        prices: pd.Series,
        regime: pd.Series,
        probabilities: pd.Series,
        allocation_type: str,
        risk_on_allocation: float = 1.0,
        risk_off_allocation: float = 0.0,
        risk_free_rate: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict]:
        """
        Simulate strategy with configurable allocations.

        Args:
            prices: BTC price series
            regime: Filtered regime series (with 21-day holding period)
            probabilities: Risk-On probability series
            allocation_type: 'binary', 'gradual', or 'dynamic'
            risk_on_allocation: BTC allocation when Risk-On (for gradual)
            risk_off_allocation: BTC allocation when Risk-Off (for gradual)
            risk_free_rate: Fed funds rate series

        Returns:
            Tuple of (portfolio_values, fee_log)
        """
        # Handle risk_free_rate
        if risk_free_rate is None:
            risk_free_rates = pd.Series(0.0437, index=prices.index)
        else:
            risk_free_rates = risk_free_rate

        # Initialize
        portfolio_value = pd.Series(index=prices.index, dtype=float)
        cash = self.initial_capital
        btc_amount = 0.0
        last_trade_idx = 0

        # Fee tracking
        fee_log = {
            'tx_fees': [],
            'slippage_fees': [],
            'pool_fees': [],
            'total_fees': []
        }

        for i in range(len(prices)):
            current_price = prices.iloc[i]
            current_regime = regime.iloc[i]
            prob = probabilities.iloc[i]

            # Compound cash at risk-free rate (before any trading)
            if i > 0 and cash > 0:
                annual_rate = risk_free_rates.iloc[i] / 100.0
                daily_rf_rate = (1 + annual_rate) ** (1/365) - 1
                cash = cash * (1 + daily_rf_rate)

            # Calculate current portfolio value before rebalancing
            current_portfolio_value = cash + (btc_amount * current_price)

            # Determine target BTC allocation
            if allocation_type == 'binary':
                target_btc_pct = 1.0 if current_regime == 1 else 0.0
            elif allocation_type == 'gradual':
                target_btc_pct = risk_on_allocation if current_regime == 1 else risk_off_allocation
            elif allocation_type == 'dynamic':
                target_btc_pct = prob  # Use probability directly
            else:
                raise ValueError(f"Unknown allocation type: {allocation_type}")

            # Calculate current BTC allocation
            current_btc_pct = (btc_amount * current_price) / current_portfolio_value if current_portfolio_value > 0 else 0

            # Calculate rebalance amount (only if allocation changed)
            allocation_change = abs(target_btc_pct - current_btc_pct)

            # Only rebalance if allocation changed significantly (avoid tiny trades)
            if i == 0 or allocation_change > 0.01:  # 1% minimum change
                # Calculate traded value
                traded_value = allocation_change * current_portfolio_value

                # Apply fees only if trading
                if traded_value > 10:  # Minimum $10 trade to avoid dust
                    slippage_loss = traded_value * self.slippage_decimal
                    pool_fee_loss = traded_value * self.pool_fee_decimal
                    tx_fee = self.tx_fee_usd
                    total_cost = slippage_loss + pool_fee_loss + tx_fee

                    # Track fees
                    fee_log['tx_fees'].append((tx_fee, prices.index[i]))
                    fee_log['slippage_fees'].append((slippage_loss, prices.index[i]))
                    fee_log['pool_fees'].append((pool_fee_loss, prices.index[i]))
                    fee_log['total_fees'].append((total_cost, prices.index[i]))

                    # Deduct fees from portfolio
                    current_portfolio_value -= total_cost

                # Execute rebalance
                target_btc_value = target_btc_pct * current_portfolio_value
                target_cash_value = (1 - target_btc_pct) * current_portfolio_value

                btc_amount = target_btc_value / current_price
                cash = target_cash_value

                last_trade_idx = i

            # Record portfolio value
            portfolio_value.iloc[i] = cash + (btc_amount * current_price)

        return portfolio_value, fee_log


def main():
    print("=" * 70)
    print("ALLOCATION STRATEGY COMPARISON")
    print("=" * 70)

    # Load model and data
    print("\n1. Loading model and data...")
    classifier = MacroRegimeClassifier()
    classifier.load('models/regime_classifier.pkl')

    pipeline = CombinedDataPipeline()
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    df = pipeline.fetch_combined_data(start_date=start_date)
    df = pipeline.create_risk_labels(df, method='btc_returns')

    # Prepare data
    print("\n2. Preparing backtest data...")
    df_clean = df.dropna()
    X = df_clean.drop(columns=['risk_regime'])

    predictions = classifier.predict(X)
    probabilities = classifier.predict(X, return_proba=True)

    backtest_df = pd.DataFrame({
        'btc_price': df_clean['btc_price'],
        'actual_regime': df_clean['risk_regime'],
        'predicted_regime': predictions,
        'risk_on_probability': probabilities,
        'fed_funds': df_clean['fed_funds']
    }, index=df_clean.index)

    # Apply 21-day holding period filter
    print("\n3. Applying 21-day holding period filter...")
    filtered_regime = pd.Series(index=backtest_df.index, dtype=int)
    current_regime = backtest_df['predicted_regime'].iloc[0]
    last_trade_date = backtest_df.index[0]

    for i, (idx, row) in enumerate(backtest_df.iterrows()):
        predicted = row['predicted_regime']
        days_since_trade = (idx - last_trade_date).days

        if i == 0:
            filtered_regime.iloc[i] = current_regime
        elif predicted != current_regime and days_since_trade >= 21:
            current_regime = predicted
            last_trade_date = idx
            filtered_regime.iloc[i] = current_regime
        else:
            filtered_regime.iloc[i] = current_regime

    backtest_df['filtered_regime'] = filtered_regime

    # Initialize backtester
    backtester = AllocationBacktester(initial_capital=10000, min_holding_days=21)

    # Test strategies
    results = []

    print("\n" + "=" * 70)
    print("TESTING ALLOCATION STRATEGIES")
    print("=" * 70)

    # 1. Binary (current approach)
    print("\n1. BINARY (100/0)...")
    portfolio, fees = backtester.simulate_allocation_strategy(
        backtest_df['btc_price'],
        backtest_df['filtered_regime'],
        backtest_df['risk_on_probability'],
        allocation_type='binary',
        risk_free_rate=backtest_df['fed_funds']
    )
    metrics = calculate_all_metrics(portfolio, 0.0437)
    metrics['total_fees'] = sum(f[0] for f in fees['total_fees'])
    metrics['num_swaps'] = len(fees['total_fees'])

    results.append({
        'strategy': 'Binary (100/0)',
        'allocation_type': 'binary',
        'risk_on': 1.0,
        'risk_off': 0.0,
        'metrics': metrics,
        'portfolio': portfolio
    })
    print(f"   CAGR: {metrics['cagr']*100:.2f}% | Sharpe: {metrics['sharpe_ratio']:.3f} | "
          f"Swaps: {metrics['num_swaps']} | Fees: ${metrics['total_fees']:.0f}")

    # 2. Gradual allocations
    gradual_configs = [
        (0.80, 0.20, "80/20"),
        (0.75, 0.25, "75/25"),
        (0.70, 0.30, "70/30"),
        (0.65, 0.35, "65/35"),
    ]

    print("\n2. GRADUAL ALLOCATIONS...")
    for risk_on, risk_off, name in gradual_configs:
        portfolio, fees = backtester.simulate_allocation_strategy(
            backtest_df['btc_price'],
            backtest_df['filtered_regime'],
            backtest_df['risk_on_probability'],
            allocation_type='gradual',
            risk_on_allocation=risk_on,
            risk_off_allocation=risk_off,
            risk_free_rate=backtest_df['fed_funds']
        )
        metrics = calculate_all_metrics(portfolio, 0.0437)
        metrics['total_fees'] = sum(f[0] for f in fees['total_fees'])
        metrics['num_swaps'] = len(fees['total_fees'])

        results.append({
            'strategy': f'Gradual ({name})',
            'allocation_type': 'gradual',
            'risk_on': risk_on,
            'risk_off': risk_off,
            'metrics': metrics,
            'portfolio': portfolio
        })
        print(f"   {name}: CAGR: {metrics['cagr']*100:.2f}% | Sharpe: {metrics['sharpe_ratio']:.3f} | "
              f"Swaps: {metrics['num_swaps']} | Fees: ${metrics['total_fees']:.0f}")

    # 3. Dynamic (probability-based)
    print("\n3. DYNAMIC (Probability-Based)...")
    portfolio, fees = backtester.simulate_allocation_strategy(
        backtest_df['btc_price'],
        backtest_df['filtered_regime'],
        backtest_df['risk_on_probability'],
        allocation_type='dynamic',
        risk_free_rate=backtest_df['fed_funds']
    )
    metrics = calculate_all_metrics(portfolio, 0.0437)
    metrics['total_fees'] = sum(f[0] for f in fees['total_fees'])
    metrics['num_swaps'] = len(fees['total_fees'])

    results.append({
        'strategy': 'Dynamic (Probability)',
        'allocation_type': 'dynamic',
        'risk_on': None,
        'risk_off': None,
        'metrics': metrics,
        'portfolio': portfolio
    })
    print(f"   CAGR: {metrics['cagr']*100:.2f}% | Sharpe: {metrics['sharpe_ratio']:.3f} | "
          f"Swaps: {metrics['num_swaps']} | Fees: ${metrics['total_fees']:.0f}")

    # Find best strategies
    print("\n" + "=" * 70)
    print("BEST STRATEGIES BY METRIC")
    print("=" * 70)

    best_cagr = max(results, key=lambda x: x['metrics']['cagr'])
    print(f"\nHighest CAGR: {best_cagr['strategy']}")
    print(f"  CAGR: {best_cagr['metrics']['cagr']*100:.2f}%")
    print(f"  Sharpe: {best_cagr['metrics']['sharpe_ratio']:.3f}")
    print(f"  Swaps: {best_cagr['metrics']['num_swaps']}")
    print(f"  Fees: ${best_cagr['metrics']['total_fees']:.2f}")
    print(f"  Final Value: ${best_cagr['metrics']['end_value']:,.2f}")

    best_sharpe = max(results, key=lambda x: x['metrics']['sharpe_ratio'])
    print(f"\nBest Sharpe: {best_sharpe['strategy']}")
    print(f"  CAGR: {best_sharpe['metrics']['cagr']*100:.2f}%")
    print(f"  Sharpe: {best_sharpe['metrics']['sharpe_ratio']:.3f}")
    print(f"  Swaps: {best_sharpe['metrics']['num_swaps']}")
    print(f"  Fees: ${best_sharpe['metrics']['total_fees']:.2f}")
    print(f"  Final Value: ${best_sharpe['metrics']['end_value']:,.2f}")

    lowest_fees = min(results, key=lambda x: x['metrics']['total_fees'])
    print(f"\nLowest Fees: {lowest_fees['strategy']}")
    print(f"  CAGR: {lowest_fees['metrics']['cagr']*100:.2f}%")
    print(f"  Sharpe: {lowest_fees['metrics']['sharpe_ratio']:.3f}")
    print(f"  Swaps: {lowest_fees['metrics']['num_swaps']}")
    print(f"  Fees: ${lowest_fees['metrics']['total_fees']:.2f}")
    print(f"  Final Value: ${lowest_fees['metrics']['end_value']:,.2f}")

    # Comparison table
    print("\n" + "=" * 70)
    print("FULL COMPARISON TABLE")
    print("=" * 70)

    comparison_data = []
    for r in results:
        m = r['metrics']
        comparison_data.append({
            'strategy': r['strategy'],
            'cagr': m['cagr'] * 100,
            'sharpe': m['sharpe_ratio'],
            'sortino': m['sortino_ratio'],
            'max_dd': m['max_drawdown'] * 100,
            'swaps': m['num_swaps'],
            'fees': m['total_fees'],
            'final_value': m['end_value']
        })

    comp_df = pd.DataFrame(comparison_data)
    print("\n" + comp_df.to_string(index=False))

    # Add Buy & Hold baseline
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    # Calculate buy & hold (with initial fee)
    buy_hold_initial = 10000
    initial_fee = buy_hold_initial * 0.013 + 5  # 1.3% + $5
    btc_bought = (buy_hold_initial - initial_fee) / backtest_df['btc_price'].iloc[0]
    buy_hold_portfolio = btc_bought * backtest_df['btc_price']
    buy_hold_metrics = calculate_all_metrics(buy_hold_portfolio, 0.0437)

    print(f"\nBuy & Hold:")
    print(f"  CAGR: {buy_hold_metrics['cagr']*100:.2f}%")
    print(f"  Sharpe: {buy_hold_metrics['sharpe_ratio']:.3f}")
    print(f"  Max DD: {buy_hold_metrics['max_drawdown']*100:.2f}%")
    print(f"  Fees: $135.00")
    print(f"  Final Value: ${buy_hold_metrics['end_value']:,.2f}")

    print(f"\nBest Strategy ({best_cagr['strategy']}):")
    print(f"  CAGR: {best_cagr['metrics']['cagr']*100:.2f}% "
          f"({best_cagr['metrics']['cagr']*100 - buy_hold_metrics['cagr']*100:+.2f}% vs B&H)")
    print(f"  Sharpe: {best_cagr['metrics']['sharpe_ratio']:.3f} "
          f"({best_cagr['metrics']['sharpe_ratio'] - buy_hold_metrics['sharpe_ratio']:+.3f} vs B&H)")
    print(f"  Max DD: {best_cagr['metrics']['max_drawdown']*100:.2f}% "
          f"({best_cagr['metrics']['max_drawdown']*100 - buy_hold_metrics['max_drawdown']*100:+.2f}% vs B&H)")
    print(f"  Fees: ${best_cagr['metrics']['total_fees']:.2f}")
    print(f"  Final Value: ${best_cagr['metrics']['end_value']:,.2f}")

    print("\n" + "=" * 70)
    print("ALLOCATION STRATEGY TESTING COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
