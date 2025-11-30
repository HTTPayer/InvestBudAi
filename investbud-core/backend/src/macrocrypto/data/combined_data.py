"""
Combined data pipeline: merges FRED macro data with BTC momentum indicators.
"""
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta

from .fred_data import FREDDataFetcher
from .btc_data import BTCDataFetcher


class CombinedDataPipeline:
    """Combines macro and crypto data for regime classification."""

    def __init__(self):
        """Initialize data fetchers."""
        self.fred_fetcher = FREDDataFetcher()
        self.btc_fetcher = BTCDataFetcher()

    def fetch_combined_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resample: str = 'D'
    ) -> pd.DataFrame:
        """
        Fetch and merge FRED macro data with BTC momentum indicators.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            resample: Resampling frequency ('D' for daily)

        Returns:
            Merged DataFrame with macro + crypto features
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print("=" * 60)
        print("FETCHING COMBINED MACRO + CRYPTO DATA")
        print("=" * 60)

        # Fetch FRED macro data
        print("\n1. Fetching macro data from FRED...")
        macro_df = self.fred_fetcher.fetch_all_indicators(
            start_date=start_date,
            end_date=end_date,
            resample=resample
        )
        macro_df = self.fred_fetcher.calculate_macro_features(macro_df)

        # Fetch BTC data
        print("\n2. Fetching BTC data...")
        btc_df = self.btc_fetcher.get_btc_features(
            start_date=start_date,
            end_date=end_date
        )

        # Select relevant BTC columns (avoid too many features)
        btc_cols = [
            'close', 'returns_1d', 'returns_7d', 'returns_30d', 'returns_60d',
            'rsi', 'drawdown', 'volatility_30d', 'price_vs_ma_30',
            'price_vs_ma_200', 'ma_30_above_200'
        ]
        btc_df = btc_df[[col for col in btc_cols if col in btc_df.columns]]

        # Rename BTC columns with 'btc_' prefix
        btc_df.columns = [f'btc_{col}' if col != 'close' else 'btc_price' for col in btc_df.columns]

        # Normalize timezones - convert both to timezone-naive for merging
        if btc_df.index.tz is not None:
            btc_df.index = btc_df.index.tz_localize(None)
        if macro_df.index.tz is not None:
            macro_df.index = macro_df.index.tz_localize(None)

        # Merge on date index
        print("\n3. Merging datasets...")
        # Forward-fill FRED data first (fills gaps before weekend/holiday alignment)
        macro_df = macro_df.ffill()
        
        # Use left join with BTC as base to keep all BTC trading days (24/7)
        # FRED data will have NaNs on weekends/holidays, filled below
        combined_df = btc_df.join(macro_df, how='left')

        # Drop rows with too many NaN values (early data may be incomplete)
        nan_threshold = 0.3  # Drop rows with >30% NaN
        combined_df = combined_df.dropna(thresh=int(len(combined_df.columns) * (1 - nan_threshold)))

        # Forward fill remaining NaNs
        combined_df = combined_df.ffill()

        print(f"\n[OK] Combined dataset ready:")
        print(f"  - Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"  - Rows: {len(combined_df)}")
        print(f"  - Features: {len(combined_df.columns)}")
        print(f"  - Macro features: {len([c for c in combined_df.columns if not c.startswith('btc_')])}")
        print(f"  - BTC features: {len([c for c in combined_df.columns if c.startswith('btc_')])}")

        return combined_df

    def create_risk_labels(
        self,
        df: pd.DataFrame,
        method: str = 'btc_returns',
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Create binary Risk-On / Risk-Off labels.

        Args:
            df: Combined dataframe
            method: Labeling method:
                - 'btc_returns': Based on BTC 30-day returns
                - 'composite': Based on multiple indicators
                - 'recession': Based on USREC flag + BTC drawdown
            threshold: Threshold for binary classification (default: 0.0)

        Returns:
            DataFrame with 'risk_regime' column (1 = Risk-On, 0 = Risk-Off)
        """
        result = df.copy()

        if method == 'btc_returns':
            # Simple: Risk-On if BTC 30-day returns > threshold
            result['risk_regime'] = (result['btc_returns_30d'] > threshold).astype(int)

        elif method == 'composite':
            # Composite: Multiple conditions
            conditions = []

            # BTC momentum positive
            if 'btc_returns_30d' in result.columns:
                conditions.append(result['btc_returns_30d'] > 0)

            # Not in recession
            if 'recession' in result.columns:
                conditions.append(result['recession'] == 0)

            # Low VIX
            if 'vix' in result.columns:
                conditions.append(result['vix'] < 25)

            # BTC not in deep drawdown
            if 'btc_drawdown' in result.columns:
                conditions.append(result['btc_drawdown'] > -0.3)

            # Risk-On if majority of conditions are true
            if conditions:
                result['risk_regime'] = (sum(conditions) > len(conditions) / 2).astype(int)
            else:
                raise ValueError("No valid conditions for composite method")

        elif method == 'recession':
            # Based on recession flag and BTC drawdown
            risk_off = (
                (result.get('recession', 0) == 1) |
                (result.get('btc_drawdown', 0) < -0.3)
            )
            result['risk_regime'] = (~risk_off).astype(int)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate regime statistics
        risk_on_pct = result['risk_regime'].mean() * 100
        print(f"\n[OK] Risk labels created ({method} method):")
        print(f"  - Risk-On: {risk_on_pct:.1f}%")
        print(f"  - Risk-Off: {100 - risk_on_pct:.1f}%")

        return result


def main():
    """Test the combined data pipeline."""
    pipeline = CombinedDataPipeline()

    # Fetch 5 years of data
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

    # Get combined data
    df = pipeline.fetch_combined_data(start_date=start_date)

    print("\n" + "=" * 60)
    print("SAMPLE DATA")
    print("=" * 60)
    print(df.head())

    # Create risk labels using different methods
    print("\n" + "=" * 60)
    print("CREATING RISK LABELS")
    print("=" * 60)

    df_btc = pipeline.create_risk_labels(df, method='btc_returns')
    print("\nSample with BTC returns method:")
    print(df_btc[['btc_price', 'btc_returns_30d', 'recession', 'risk_regime']].tail(10))

    df_composite = pipeline.create_risk_labels(df, method='composite')
    print("\nSample with composite method:")
    print(df_composite[['btc_price', 'btc_returns_30d', 'recession', 'vix', 'risk_regime']].tail(10))

    # Show feature names
    print("\n" + "=" * 60)
    print("AVAILABLE FEATURES")
    print("=" * 60)
    print("\nMacro features:")
    macro_cols = [c for c in df.columns if not c.startswith('btc_')]
    for col in sorted(macro_cols):
        print(f"  - {col}")

    print("\nBTC features:")
    btc_cols = [c for c in df.columns if c.startswith('btc_')]
    for col in sorted(btc_cols):
        print(f"  - {col}")


if __name__ == '__main__':
    main()
