"""
Run backtest for MacroCrypto regime-based trading strategies.
Run with: uv run python run_backtest.py
"""
from src.macrocrypto.models import RegimeBacktester, MacroRegimeClassifier
from src.macrocrypto.data import CombinedDataPipeline
from datetime import datetime, timedelta


def main():
    print("=" * 70)
    print("MacroCrypto Backtesting Framework")
    print("=" * 70)

    # Initialize
    backtester = RegimeBacktester(initial_capital=10000)
    classifier = MacroRegimeClassifier()
    pipeline = CombinedDataPipeline()

    # Load trained model
    print("\n1. Loading trained model...")
    try:
        classifier.load('models/regime_classifier.pkl')
    except FileNotFoundError:
        print("[ERROR] Model not found. Please run 'uv run python train_classifier.py' first.")
        return

    # Fetch data
    print("\n2. Fetching data...")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    df = pipeline.fetch_combined_data(start_date=start_date)
    df = pipeline.create_risk_labels(df, method='btc_returns')

    # Prepare backtest data
    print("\n3. Preparing backtest data...")
    backtest_df = backtester.prepare_backtest_data(df, classifier)

    # Apply 21-day holding period filter (optimal from grid search)
    print("\n4. Applying 21-day holding period filter...")
    backtest_df = backtester.apply_holding_period_filter(backtest_df, min_days=21)

    print(f"[OK] Backtest data ready: {len(backtest_df)} periods")
    print(f"Date range: {backtest_df.index[0]} to {backtest_df.index[-1]}")

    # Show actual vs predicted regime accuracy
    accuracy = (backtest_df['actual_regime'] == backtest_df['predicted_regime']).mean()
    print(f"Model Accuracy on backtest period: {accuracy*100:.1f}%")

    # Count regime switches before and after filter
    raw_switches = (backtest_df['predicted_regime'].diff() != 0).sum()
    filtered_switches = (backtest_df['filtered_regime'].diff() != 0).sum()
    print(f"Regime switches: {raw_switches} unfiltered -> {filtered_switches} filtered ({filtered_switches - raw_switches:+d})")

    # Run backtest
    print("\n5. Running backtest...")
    results = backtester.run_backtest(backtest_df, risk_free_rate=0.02)

    # Save results for /historical endpoint
    print("\n6. Saving results...")
    backtester.save_results('models/backtest_results.pkl')

    # Print results
    print("\n7. Analyzing results...")
    backtester.print_results(risk_free_rate=0.02)

    # Trade log
    print("\n8. Generating trade log...")
    trade_log = backtester.get_trade_log(backtest_df)

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    metrics = backtester.calculate_metrics()

    model_cagr = metrics['model']['cagr'] * 100
    buy_hold_cagr = metrics['buy_and_hold']['cagr'] * 100
    model_sharpe = metrics['model']['sharpe_ratio']
    buy_hold_sharpe = metrics['buy_and_hold']['sharpe_ratio']
    model_max_dd = metrics['model']['max_drawdown'] * 100
    buy_hold_max_dd = metrics['buy_and_hold']['max_drawdown'] * 100

    print(f"\nModel-Based vs Buy & Hold:")
    print(f"  CAGR:        {model_cagr:>7.2f}% vs {buy_hold_cagr:>7.2f}%  ({model_cagr - buy_hold_cagr:+.2f}%)")
    print(f"  Sharpe:      {model_sharpe:>7.3f} vs {buy_hold_sharpe:>7.3f}  ({model_sharpe - buy_hold_sharpe:+.3f})")
    print(f"  Max Drawdown:{model_max_dd:>7.2f}% vs {buy_hold_max_dd:>7.2f}%  ({model_max_dd - buy_hold_max_dd:+.2f}%)")

    if model_sharpe > buy_hold_sharpe:
        print("\n[+] Model-based strategy outperforms buy & hold on risk-adjusted returns!")
    else:
        print("\n[-] Buy & hold outperforms model-based strategy")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()
