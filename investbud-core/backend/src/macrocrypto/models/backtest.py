"""
Backtesting framework for MacroCrypto regime-based trading strategies.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from ..data import CombinedDataPipeline
from ..utils.metrics import calculate_all_metrics, print_metrics_report, compare_metrics
from ..utils.advanced_metrics import calculate_all_advanced_metrics
from .regime_classifier import MacroRegimeClassifier


class RegimeBacktester:
    """
    Backtest trading strategies based on macro regime predictions.

    Strategies:
    - Buy and Hold: Always 100% BTC
    - Always Risk-On: Always 100% BTC
    - Always Risk-Off: Always 0% BTC (cash)
    - Model-Based: Follow model predictions (BTC in Risk-On, cash in Risk-Off)
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        slippage_decimal: float = 0.01,
        pool_fee_decimal: float = 0.003,
        tx_fee_usd: float = 5.0
    ):
        """
        Initialize the backtester.

        Args:
            initial_capital: Starting portfolio value in USD
            slippage_decimal: Slippage as decimal (default 0.01 = 1%)
            pool_fee_decimal: Pool/swap fee as decimal (default 0.003 = 0.3%)
            tx_fee_usd: Fixed transaction fee in USD per swap (default $5)
        """
        self.initial_capital = initial_capital
        self.slippage_decimal = slippage_decimal
        self.pool_fee_decimal = pool_fee_decimal
        self.tx_fee_usd = tx_fee_usd
        self.results = {}
        self.fee_logs = {}  # Track fees for each strategy

    def prepare_backtest_data(
        self,
        df: pd.DataFrame,
        classifier: MacroRegimeClassifier
    ) -> pd.DataFrame:
        """
        Prepare data for backtesting with model predictions.

        Args:
            df: Combined dataframe with features and actual labels
            classifier: Trained classifier

        Returns:
            DataFrame with predictions and BTC prices
        """
        # Drop NaN rows
        df_clean = df.dropna()

        # Get features (drop target)
        X = df_clean.drop(columns=['risk_regime'])

        # Get model predictions
        predictions = classifier.predict(X)
        probabilities = classifier.predict(X, return_proba=True)

        # Create backtest dataframe
        backtest_df = pd.DataFrame({
            'btc_price': df_clean['btc_price'],
            'actual_regime': df_clean['risk_regime'],
            'predicted_regime': predictions,
            'risk_on_probability': probabilities,
            'fed_funds': df_clean['fed_funds']  # Actual risk-free rate
        }, index=df_clean.index)

        return backtest_df

    def apply_holding_period_filter(
        self,
        backtest_df: pd.DataFrame,
        min_days: int = 21
    ) -> pd.DataFrame:
        """
        Apply minimum holding period filter to reduce trading frequency.

        Only allows regime switches if min_days have passed since last trade.
        This dramatically reduces fees and improves performance.

        Args:
            backtest_df: DataFrame with predicted_regime column
            min_days: Minimum days to hold position before switching (default 21)

        Returns:
            DataFrame with filtered_regime column added
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

        # Add filtered regime to dataframe
        backtest_df = backtest_df.copy()
        backtest_df['filtered_regime'] = filtered_regime

        return backtest_df

    def simulate_strategy(
        self,
        prices: pd.Series,
        regime: pd.Series,
        strategy: str = 'model',
        risk_free_rate: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict]:
        """
        Simulate a trading strategy and return portfolio values.

        Args:
            prices: BTC prices
            regime: Risk regime signal (1 = Risk-On, 0 = Risk-Off)
            strategy: Strategy type:
                - 'buy_and_hold': Always 100% BTC
                - 'always_risk_on': Always 100% BTC
                - 'always_risk_off': Always 0% BTC (cash earning risk-free rate)
                - 'model': Follow regime signals
            risk_free_rate: Annual risk-free rate for cash positions (Series or scalar)
                          If None, uses 2% constant rate

        Returns:
            Tuple of (portfolio_values, fee_log)
        """
        # Handle risk_free_rate input
        if risk_free_rate is None:
            # Default to 2% constant
            risk_free_rates = pd.Series(0.02, index=prices.index)
        elif isinstance(risk_free_rate, (int, float)):
            # Scalar - convert to Series
            risk_free_rates = pd.Series(risk_free_rate, index=prices.index)
        else:
            # Already a Series
            risk_free_rates = risk_free_rate

        # Initialize portfolio
        portfolio_value = pd.Series(index=prices.index, dtype=float)
        portfolio_value.iloc[0] = self.initial_capital

        # Initialize fee tracking
        fee_log = {
            'tx_fees': [],
            'slippage_fees': [],
            'pool_fees': [],
            'total_fees': []
        }

        # Determine allocation based on strategy
        if strategy == 'buy_and_hold' or strategy == 'always_risk_on':
            # Buy BTC on day 1 and hold
            # Apply fees on initial purchase
            traded_value = self.initial_capital
            slippage_loss = traded_value * self.slippage_decimal
            pool_fee_loss = traded_value * self.pool_fee_decimal
            tx_fee = self.tx_fee_usd
            total_cost = tx_fee + slippage_loss + pool_fee_loss

            # Track fees
            fee_log['tx_fees'].append((tx_fee, prices.index[0]))
            fee_log['slippage_fees'].append((slippage_loss, prices.index[0]))
            fee_log['pool_fees'].append((pool_fee_loss, prices.index[0]))
            fee_log['total_fees'].append((total_cost, prices.index[0]))

            # Buy with remaining capital after fees
            btc_amount = (self.initial_capital - total_cost) / prices.iloc[0]
            portfolio_value = btc_amount * prices

        elif strategy == 'always_risk_off':
            # Stay in cash earning risk-free rate
            # Compound daily using actual rates
            # No fees since no swaps
            cash = self.initial_capital
            for i in range(len(prices)):
                if i > 0:
                    # Get daily risk-free rate (convert annual to daily)
                    annual_rate = risk_free_rates.iloc[i] / 100.0  # Convert from percentage
                    daily_rf_rate = (1 + annual_rate) ** (1/365) - 1
                    # Add daily interest
                    cash = cash * (1 + daily_rf_rate)
                portfolio_value.iloc[i] = cash

        elif strategy == 'model':
            # Follow regime signals with fees on each swap
            cash = self.initial_capital
            btc_amount = 0.0

            for i in range(len(prices)):
                current_price = prices.iloc[i]
                current_regime = regime.iloc[i]

                # Get previous regime
                prev_regime = regime.iloc[i-1] if i > 0 else 0

                # Compound cash position if in Risk-Off (before rebalancing)
                if i > 0 and cash > 0:
                    # Use actual Fed Funds rate for this day
                    annual_rate = risk_free_rates.iloc[i] / 100.0  # Convert from percentage
                    daily_rf_rate = (1 + annual_rate) ** (1/365) - 1
                    cash = cash * (1 + daily_rf_rate)

                # Rebalance on regime change
                if i == 0:
                    # Initial position based on first regime
                    if current_regime == 1:  # Risk-On
                        # Apply fees on initial BTC purchase
                        traded_value = cash
                        slippage_loss = traded_value * self.slippage_decimal
                        pool_fee_loss = traded_value * self.pool_fee_decimal
                        tx_fee = self.tx_fee_usd
                        total_cost = slippage_loss + pool_fee_loss + tx_fee

                        # Track fees
                        fee_log['tx_fees'].append((tx_fee, prices.index[i]))
                        fee_log['slippage_fees'].append((slippage_loss, prices.index[i]))
                        fee_log['pool_fees'].append((pool_fee_loss, prices.index[i]))
                        fee_log['total_fees'].append((total_cost, prices.index[i]))

                        # Buy BTC with remaining cash after fees
                        cash_after_fees = cash - total_cost
                        btc_amount = cash_after_fees / current_price
                        cash = 0
                    else:  # Risk-Off
                        # Stay in cash, no fees
                        btc_amount = 0
                        cash = self.initial_capital

                elif current_regime != prev_regime:
                    # Regime changed - rebalance with fees
                    if current_regime == 1:  # Switch to Risk-On (buy BTC)
                        # Apply fees on BTC purchase
                        traded_value = cash
                        slippage_loss = traded_value * self.slippage_decimal
                        pool_fee_loss = traded_value * self.pool_fee_decimal
                        tx_fee = self.tx_fee_usd
                        total_cost = slippage_loss + pool_fee_loss + tx_fee

                        # Track fees
                        fee_log['tx_fees'].append((tx_fee, prices.index[i]))
                        fee_log['slippage_fees'].append((slippage_loss, prices.index[i]))
                        fee_log['pool_fees'].append((pool_fee_loss, prices.index[i]))
                        fee_log['total_fees'].append((total_cost, prices.index[i]))

                        # Buy BTC with remaining cash after fees
                        cash_after_fees = cash - total_cost
                        btc_amount = cash_after_fees / current_price
                        cash = 0

                    else:  # Switch to Risk-Off (sell BTC)
                        # Apply fees on BTC sale
                        traded_value = btc_amount * current_price
                        slippage_loss = traded_value * self.slippage_decimal
                        pool_fee_loss = traded_value * self.pool_fee_decimal
                        tx_fee = self.tx_fee_usd
                        total_cost = slippage_loss + pool_fee_loss + tx_fee

                        # Track fees
                        fee_log['tx_fees'].append((tx_fee, prices.index[i]))
                        fee_log['slippage_fees'].append((slippage_loss, prices.index[i]))
                        fee_log['pool_fees'].append((pool_fee_loss, prices.index[i]))
                        fee_log['total_fees'].append((total_cost, prices.index[i]))

                        # Sell BTC to cash and deduct fees
                        cash = traded_value - total_cost
                        btc_amount = 0

                # Calculate portfolio value
                portfolio_value.iloc[i] = cash + (btc_amount * current_price)

        return portfolio_value, fee_log

    def run_backtest(
        self,
        backtest_df: pd.DataFrame,
        strategies: Optional[list] = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, pd.Series]:
        """
        Run backtest for multiple strategies.

        Args:
            backtest_df: DataFrame with prices and regime predictions
            strategies: List of strategies to test (default: all)
            risk_free_rate: Annual risk-free rate for cash positions (used if fed_funds not in df)

        Returns:
            Dictionary of {strategy_name: portfolio_values}
        """
        if strategies is None:
            strategies = ['buy_and_hold', 'always_risk_on', 'always_risk_off', 'model']

        # Use actual Fed Funds data if available
        if 'fed_funds' in backtest_df.columns:
            rf_rates = backtest_df['fed_funds']
            rf_avg = rf_rates.mean()
            rf_min = rf_rates.min()
            rf_max = rf_rates.max()
            rf_display = f"{rf_avg:.2f}% avg (range: {rf_min:.2f}%-{rf_max:.2f}%)"
        else:
            rf_rates = risk_free_rate
            rf_display = f"{risk_free_rate*100:.1f}% (constant)"

        # Check if filtered regime is available
        use_filtered = 'filtered_regime' in backtest_df.columns

        print("\n" + "=" * 70)
        print("RUNNING BACKTEST")
        print("=" * 70)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Date Range: {backtest_df.index[0]} to {backtest_df.index[-1]}")
        print(f"Total Periods: {len(backtest_df)}")
        print(f"Strategies: {', '.join(strategies)}")
        print(f"Risk-Free Rate: {rf_display}")
        print(f"\nFee Structure:")
        print(f"  Slippage: {self.slippage_decimal*100:.1f}%")
        print(f"  Pool Fee: {self.pool_fee_decimal*100:.2f}%")
        print(f"  TX Fee: ${self.tx_fee_usd:.2f} per swap")

        if use_filtered:
            print(f"\nTrading Filter: 21-day minimum holding period")
            print(f"  (Reduces excessive trading and fee drag)")

        results = {}

        for strategy in strategies:
            print(f"\nSimulating {strategy}...")

            if strategy == 'model':
                # Use filtered regime if available (21-day holding period)
                regime = backtest_df['filtered_regime'] if use_filtered else backtest_df['predicted_regime']
            elif strategy == 'always_risk_on':
                regime = pd.Series(1, index=backtest_df.index)
            elif strategy == 'always_risk_off':
                regime = pd.Series(0, index=backtest_df.index)
            else:  # buy_and_hold
                regime = pd.Series(1, index=backtest_df.index)

            portfolio_values, fee_log = self.simulate_strategy(
                backtest_df['btc_price'],
                regime,
                strategy,
                rf_rates
            )

            results[strategy] = portfolio_values
            self.fee_logs[strategy] = fee_log

        self.results = results
        return results

    def calculate_metrics(
        self,
        risk_free_rate: float = 0.02
    ) -> Dict[str, dict]:
        """
        Calculate performance metrics for all strategies.

        Args:
            risk_free_rate: Annual risk-free rate

        Returns:
            Dictionary of {strategy_name: metrics_dict}
        """
        if not self.results:
            raise ValueError("No backtest results. Run run_backtest() first.")

        all_metrics = {}

        # Get buy_and_hold as benchmark (or use first strategy if not available)
        benchmark_key = 'buy_and_hold' if 'buy_and_hold' in self.results else list(self.results.keys())[0]
        benchmark_values = self.results[benchmark_key]

        for strategy, portfolio_values in self.results.items():
            # Calculate basic metrics
            metrics = calculate_all_metrics(portfolio_values, risk_free_rate)

            # Add fee information
            if strategy in self.fee_logs:
                fee_log = self.fee_logs[strategy]
                total_tx_fees = sum(f[0] for f in fee_log['tx_fees'])
                total_slippage = sum(f[0] for f in fee_log['slippage_fees'])
                total_pool_fees = sum(f[0] for f in fee_log['pool_fees'])
                total_all_fees = sum(f[0] for f in fee_log['total_fees'])
                num_swaps = len(fee_log['total_fees'])

                metrics['total_tx_fees'] = total_tx_fees
                metrics['total_slippage'] = total_slippage
                metrics['total_pool_fees'] = total_pool_fees
                metrics['total_fees'] = total_all_fees
                metrics['num_swaps'] = num_swaps
                metrics['avg_fee_per_swap'] = total_all_fees / num_swaps if num_swaps > 0 else 0
            else:
                metrics['total_fees'] = 0
                metrics['num_swaps'] = 0

            # Calculate advanced metrics vs benchmark (skip for benchmark itself)
            if strategy != benchmark_key:
                try:
                    advanced = calculate_all_advanced_metrics(
                        portfolio_values,
                        benchmark_values,
                        risk_free_rate
                    )
                    metrics.update(advanced)
                except Exception as e:
                    print(f"[WARNING] Could not calculate advanced metrics for {strategy}: {e}")

            all_metrics[strategy] = metrics

        return all_metrics

    def save_results(self, filepath: str = 'models/backtest_results.pkl', risk_free_rate: float = 0.02):
        """
        Save backtest results to a pickle file for /historical endpoint.

        Args:
            filepath: Path to save results
            risk_free_rate: Annual risk-free rate
        """
        import pickle
        import os

        if not self.results:
            raise ValueError("No backtest results. Run run_backtest() first.")

        # Calculate metrics
        metrics = self.calculate_metrics(risk_free_rate)

        # Prepare data for saving
        backtest_data = {
            'results': self.results,  # Dict[str, pd.Series] - portfolio values
            'metrics': metrics,  # Dict[str, dict] - performance metrics
            'initial_capital': self.initial_capital,
            'timestamp': datetime.now().isoformat()
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save to pickle
        with open(filepath, 'wb') as f:
            pickle.dump(backtest_data, f)

        print(f"[OK] Backtest results saved to {filepath}")

    def print_results(self, risk_free_rate: float = 0.02):
        """
        Print detailed backtest results for all strategies.

        Args:
            risk_free_rate: Annual risk-free rate
        """
        metrics = self.calculate_metrics(risk_free_rate)

        # Print individual strategy metrics
        for strategy, strategy_metrics in metrics.items():
            print_metrics_report(strategy_metrics, strategy_name=strategy.replace('_', ' ').title())

        # Print comparison
        compare_metrics(metrics)

    def get_trade_log(self, backtest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a trade log showing regime changes and trades.

        Args:
            backtest_df: DataFrame with prices and regime predictions

        Returns:
            DataFrame with trade log
        """
        trades = []

        regime = backtest_df['predicted_regime']
        prices = backtest_df['btc_price']

        prev_regime = None
        position = 'cash'  # Start in cash

        for i in range(len(regime)):
            current_regime = regime.iloc[i]
            current_price = prices.iloc[i]
            date = regime.index[i]

            # Check for regime change
            if prev_regime is not None and current_regime != prev_regime:
                # Regime changed
                if current_regime == 1:  # Risk-Off -> Risk-On
                    action = 'BUY'
                    position = 'btc'
                else:  # Risk-On -> Risk-Off
                    action = 'SELL'
                    position = 'cash'

                trades.append({
                    'date': date,
                    'action': action,
                    'regime': 'Risk-On' if current_regime == 1 else 'Risk-Off',
                    'price': current_price,
                    'position_after': position
                })

            prev_regime = current_regime

        trade_log = pd.DataFrame(trades)

        if not trade_log.empty:
            print("\n" + "=" * 70)
            print("TRADE LOG (Model-Based Strategy)")
            print("=" * 70)
            print(trade_log.to_string(index=False))
            print(f"\nTotal Trades: {len(trade_log)}")

        return trade_log


def main():
    """Run a complete backtest example."""
    print("=" * 70)
    print("MacroCrypto Backtesting Framework")
    print("=" * 70)

    # Initialize
    backtester = RegimeBacktester(initial_capital=10000)
    classifier = MacroRegimeClassifier()
    pipeline = CombinedDataPipeline()

    # Load trained model
    print("\n1. Loading trained model...")
    classifier.load('models/regime_classifier.pkl')

    # Fetch data
    print("\n2. Fetching data...")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    df = pipeline.fetch_combined_data(start_date=start_date)
    df = pipeline.create_risk_labels(df, method='btc_returns')

    # Prepare backtest data
    print("\n3. Preparing backtest data...")
    backtest_df = backtester.prepare_backtest_data(df, classifier)

    print(f"[OK] Backtest data ready: {len(backtest_df)} periods")
    print(f"Date range: {backtest_df.index[0]} to {backtest_df.index[-1]}")

    # Run backtest
    print("\n4. Running backtest...")
    results = backtester.run_backtest(backtest_df)

    # Print results
    print("\n5. Analyzing results...")
    backtester.print_results(risk_free_rate=0.02)

    # Trade log
    print("\n6. Generating trade log...")
    trade_log = backtester.get_trade_log(backtest_df)

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
