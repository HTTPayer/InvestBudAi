"""
Train the MacroCrypto regime classifier.
Run with: uv run python train_classifier.py
"""
from src.macrocrypto.models import MacroRegimeClassifier, RegimeBacktester
from src.macrocrypto.data import CombinedDataPipeline
from datetime import datetime, timedelta


def main():
    print("=" * 70)
    print("MacroCrypto Regime Classifier Training")
    print("=" * 70)

    # Initialize
    classifier = MacroRegimeClassifier()
    pipeline = CombinedDataPipeline()

    # Fetch data (2 years)
    print("\n1. Fetching data...")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    df = pipeline.fetch_combined_data(start_date=start_date)

    # Create labels
    print("\n2. Creating risk regime labels...")
    df = pipeline.create_risk_labels(df, method='btc_returns')

    # Prepare data
    print("\n3. Preparing data...")
    X, y = classifier.prepare_data(df)

    # Train
    print("\n4. Training classifier...")
    metrics = classifier.train(X, y, test_size=0.2)

    # Cross-validation
    print("\n4b. Running time-series cross-validation...")
    cv_metrics = classifier.cross_validate(X, y, n_splits=5)

    # Feature importance
    print("\n" + "=" * 70)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 70)
    importance = classifier.get_feature_importance(top_n=15)
    print(importance.to_string(index=False))

    # Predict current regime
    print("\n5. Testing prediction on current data...")
    current = classifier.predict_current_regime()

    # Save model
    print("\n6. Saving model...")
    model_path = 'models/regime_classifier.pkl'
    classifier.save(model_path)

    # Run backtest
    print("\n7. Running backtest...")
    backtester = RegimeBacktester(initial_capital=10000)
    backtest_df = backtester.prepare_backtest_data(df, classifier)

    # Apply 21-day holding period filter (optimal from grid search)
    backtest_df = backtester.apply_holding_period_filter(backtest_df, min_days=21)

    results = backtester.run_backtest(backtest_df)

    # Save backtest results for /historical endpoint
    print("\n8. Saving backtest results...")
    backtester.save_results('models/backtest_results.pkl')

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved to: {model_path}")
    print(f"Test Accuracy: {metrics['test']['accuracy']*100:.1f}%")
    print(f"Test F1 Score: {metrics['test']['f1']:.3f}")
    print(f"Current Regime: {current['regime']} ({current['confidence']*100:.1f}% confidence)")

    # Print backtest summary
    backtest_metrics = backtester.calculate_metrics()
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)
    for strategy in ['model', 'buy_and_hold']:
        m = backtest_metrics[strategy]
        print(f"\n{strategy.replace('_', ' ').title()}:")
        print(f"  CAGR: {m['cagr']*100:.1f}%")
        print(f"  Sharpe: {m['sharpe_ratio']:.3f}")
        print(f"  Sortino: {m['sortino_ratio']:.3f}")
        print(f"  Max Drawdown: {m['max_drawdown']*100:.1f}%")


if __name__ == '__main__':
    main()
