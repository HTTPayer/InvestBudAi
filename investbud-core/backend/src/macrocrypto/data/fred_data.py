"""
FRED API data fetcher for macroeconomic indicators.
Fetches M2, GDP, CPI, Fed Funds, Recession flags, etc.
"""
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()


class FREDDataFetcher:
    """Fetch macroeconomic data from FRED API."""

    # FRED series IDs for macro indicators
    SERIES_IDS = {
        'm2': 'M2SL',                    # M2 Money Supply
        'gdp': 'GDP',                    # GDP (Quarterly)
        'cpi': 'CPIAUCSL',               # Consumer Price Index
        'fed_funds': 'FEDFUNDS',         # Federal Funds Rate
        'recession': 'USREC',            # Recession Indicator
        'dxy': 'DTWEXBGS',               # Dollar Index
        'unemployment': 'UNRATE',        # Unemployment Rate
        'vix': 'VIXCLS',                 # VIX Volatility Index
        'treasury_10y': 'DGS10',         # 10-Year Treasury Rate
        'treasury_2y': 'DGS2',           # 2-Year Treasury Rate
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED data fetcher.

        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError(
                "FRED API key not found. Set FRED_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.fred = Fred(api_key=self.api_key)

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch a single FRED series.

        Args:
            series_id: FRED series ID (e.g., 'M2SL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            pandas Series with datetime index
        """
        return self.fred.get_series(series_id, start_date, end_date)

    def fetch_all_indicators(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resample: str = 'D'
    ) -> pd.DataFrame:
        """
        Fetch all macro indicators and combine into a single DataFrame.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format (default: 10 years ago)
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            resample: Resampling frequency ('D' for daily, 'W' for weekly, 'M' for monthly)

        Returns:
            DataFrame with all indicators, forward-filled for missing data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Fetching FRED data from {start_date} to {end_date}...")

        # Fetch all series
        data_dict = {}
        for name, series_id in self.SERIES_IDS.items():
            try:
                print(f"  Fetching {name} ({series_id})...")
                series = self.fetch_series(series_id, start_date, end_date)
                data_dict[name] = series
            except Exception as e:
                print(f"  Warning: Failed to fetch {name}: {e}")
                continue

        # Combine into DataFrame
        df = pd.DataFrame(data_dict)

        # Resample to desired frequency and forward fill
        if resample:
            df = df.resample(resample).ffill()

        # Fill any remaining NaNs with forward fill, then backward fill
        df = df.ffill().bfill()

        print(f"[OK] Fetched {len(df.columns)} indicators with {len(df)} rows")

        return df

    def calculate_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived features from raw macro data.

        Features include:
        - Growth rates (YoY, MoM)
        - Moving averages
        - Rate of change
        - Yield curve (10Y - 2Y spread)

        Args:
            df: DataFrame with raw FRED data

        Returns:
            DataFrame with additional feature columns
        """
        result = df.copy()

        # M2 growth rate (YoY)
        if 'm2' in result.columns:
            result['m2_yoy'] = result['m2'].pct_change(periods=365)
            result['m2_mom'] = result['m2'].pct_change(periods=30)

        # GDP growth rate (QoQ annualized)
        if 'gdp' in result.columns:
            result['gdp_growth'] = result['gdp'].pct_change(periods=90) * 4  # Annualized

        # CPI inflation rate (YoY)
        if 'cpi' in result.columns:
            result['cpi_yoy'] = result['cpi'].pct_change(periods=365)
            result['cpi_mom'] = result['cpi'].pct_change(periods=30)

        # Fed Funds rate change
        if 'fed_funds' in result.columns:
            result['fed_funds_change'] = result['fed_funds'].diff()
            result['fed_funds_ma_30'] = result['fed_funds'].rolling(30).mean()

        # Yield curve (10Y - 2Y spread) - inverted curve signals recession
        if 'treasury_10y' in result.columns and 'treasury_2y' in result.columns:
            result['yield_curve_spread'] = result['treasury_10y'] - result['treasury_2y']
            result['yield_curve_inverted'] = (result['yield_curve_spread'] < 0).astype(int)

        # VIX features
        if 'vix' in result.columns:
            result['vix_ma_30'] = result['vix'].rolling(30).mean()
            result['vix_elevated'] = (result['vix'] > 20).astype(int)

        # Dollar strength
        if 'dxy' in result.columns:
            result['dxy_change'] = result['dxy'].pct_change()
            result['dxy_ma_30'] = result['dxy'].rolling(30).mean()

        # Unemployment rate change
        if 'unemployment' in result.columns:
            result['unemployment_change'] = result['unemployment'].diff()

        print(f"[OK] Calculated {len(result.columns) - len(df.columns)} derived features")

        return result


def main():
    """Test the FRED data fetcher."""
    fetcher = FREDDataFetcher()

    # Fetch last 2 years of data
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    df = fetcher.fetch_all_indicators(start_date=start_date, resample='D')

    print("\nRaw macro data:")
    print(df.head())
    print(f"\nShape: {df.shape}")

    # Calculate features
    df_features = fetcher.calculate_macro_features(df)

    print("\nWith derived features:")
    print(df_features.head())
    print(f"\nShape: {df_features.shape}")
    print(f"\nColumns: {df_features.columns.tolist()}")

    # Show latest values
    print("\nLatest macro snapshot:")
    print(df_features.iloc[-1])


if __name__ == '__main__':
    main()
