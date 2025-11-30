"""
Test script for the MacroCrypto data pipeline.
Run with: uv run python test_data_pipeline.py
"""
from src.macrocrypto.data import CombinedDataPipeline
from datetime import datetime, timedelta


def main():
    print("=" * 70)
    print("MacroCrypto Data Pipeline Test")
    print("=" * 70)

    # Initialize pipeline
    pipeline = CombinedDataPipeline()

    # Fetch 2 years of data (faster for testing)
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')

    print(f"\nFetching data from {start_date} to today...")

    # Get combined data
    df = pipeline.fetch_combined_data(start_date=start_date)

    print("\n" + "=" * 70)
    print("DATA SAMPLE (First 5 rows)")
    print("=" * 70)
    print(df.head())

    print("\n" + "=" * 70)
    print("DATA SAMPLE (Last 5 rows)")
    print("=" * 70)
    print(df.tail())

    # Create risk labels
    print("\n" + "=" * 70)
    print("CREATING RISK-ON / RISK-OFF LABELS")
    print("=" * 70)

    # Test different labeling methods
    df_labeled = pipeline.create_risk_labels(df, method='btc_returns')

    # Show recent regime
    print("\nRecent risk regime (last 30 days):")
    recent = df_labeled[['btc_price', 'btc_returns_30d', 'recession', 'risk_regime']].tail(30)
    print(recent)

    # Current regime
    latest = df_labeled.iloc[-1]
    current_regime = "Risk-On [+]" if latest['risk_regime'] == 1 else "Risk-Off [-]"

    print("\n" + "=" * 70)
    print("CURRENT MACRO REGIME")
    print("=" * 70)
    print(f"Regime: {current_regime}")
    print(f"BTC Price: ${latest['btc_price']:,.2f}")
    print(f"BTC 30d Return: {latest['btc_returns_30d']*100:.2f}%")
    print(f"Fed Funds Rate: {latest.get('fed_funds', 0):.2f}%")
    print(f"CPI YoY: {latest.get('cpi_yoy', 0)*100:.2f}%")
    print(f"In Recession: {'Yes' if latest.get('recession', 0) == 1 else 'No'}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(df_labeled)}")
    print(f"Date range: {df_labeled.index.min()} to {df_labeled.index.max()}")
    print(f"Total features: {len(df_labeled.columns)}")
    print(f"Risk-On days: {(df_labeled['risk_regime']==1).sum()} ({(df_labeled['risk_regime']==1).mean()*100:.1f}%)")
    print(f"Risk-Off days: {(df_labeled['risk_regime']==0).sum()} ({(df_labeled['risk_regime']==0).mean()*100:.1f}%)")

    print("\n[OK] Data pipeline test completed successfully!")


if __name__ == '__main__':
    main()
