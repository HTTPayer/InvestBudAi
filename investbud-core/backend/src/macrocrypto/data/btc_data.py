"""
Bitcoin data fetcher with momentum indicators.
Calculates returns, RSI, drawdowns, volatility, etc.
"""
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import yfinance as yf


class BTCDataFetcher:
    """Fetch Bitcoin price data and calculate momentum indicators."""

    def __init__(self, ticker: str = 'BTC-USD'):
        """
        Initialize BTC data fetcher.

        Args:
            ticker: Yahoo Finance ticker symbol (default: BTC-USD)
        """
        self.ticker = ticker

    def fetch_price_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch BTC price data from Yahoo Finance.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Fetching BTC data from {start_date} to {end_date}...")

        btc = yf.Ticker(self.ticker)
        df = btc.history(start=start_date, end=end_date, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {self.ticker}")

        # Clean column names
        df.columns = df.columns.str.lower()

        print(f"[OK] Fetched {len(df)} rows of BTC price data")

        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return metrics.

        Args:
            df: DataFrame with price data (must have 'close' column)

        Returns:
            DataFrame with additional return columns
        """
        result = df.copy()

        # Daily returns
        result['returns_1d'] = result['close'].pct_change()

        # Multi-period returns
        result['returns_7d'] = result['close'].pct_change(periods=7)
        result['returns_30d'] = result['close'].pct_change(periods=30)
        result['returns_60d'] = result['close'].pct_change(periods=60)
        result['returns_90d'] = result['close'].pct_change(periods=90)
        result['returns_180d'] = result['close'].pct_change(periods=180)

        # Log returns
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))

        return result

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            df: DataFrame with price data
            period: RSI period (default: 14)

        Returns:
            DataFrame with RSI column
        """
        result = df.copy()

        # Calculate price changes
        delta = result['close'].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))

        # RSI overbought/oversold flags
        result['rsi_overbought'] = (result['rsi'] > 70).astype(int)
        result['rsi_oversold'] = (result['rsi'] < 30).astype(int)

        return result

    def calculate_drawdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drawdown from all-time high.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with drawdown columns
        """
        result = df.copy()

        # Calculate running maximum (all-time high)
        result['ath'] = result['close'].expanding().max()

        # Calculate drawdown percentage
        result['drawdown'] = (result['close'] - result['ath']) / result['ath']

        # Days since ATH
        ath_date = result['ath'].idxmax()
        # Calculate days since ATH for all rows
        timedelta_index = result.index - ath_date
        result['days_since_ath'] = timedelta_index.days
        # Zero out days before ATH
        result.loc[result.index <= ath_date, 'days_since_ath'] = 0

        return result

    def calculate_volatility(self, df: pd.DataFrame, windows: list = [7, 30, 90]) -> pd.DataFrame:
        """
        Calculate rolling volatility (standard deviation of returns).

        Args:
            df: DataFrame with returns data
            windows: List of window sizes for rolling calculation

        Returns:
            DataFrame with volatility columns
        """
        result = df.copy()

        if 'returns_1d' not in result.columns:
            result['returns_1d'] = result['close'].pct_change()

        for window in windows:
            result[f'volatility_{window}d'] = result['returns_1d'].rolling(window).std()

        # Annualized volatility (assuming 365 trading days)
        result['volatility_30d_annualized'] = result['volatility_30d'] * np.sqrt(365)

        return result

    def calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all momentum indicators at once.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all momentum features
        """
        print("Calculating BTC momentum indicators...")

        result = self.calculate_returns(df)
        result = self.calculate_rsi(result)
        result = self.calculate_drawdown(result)
        result = self.calculate_volatility(result)

        # Moving averages
        result['ma_7'] = result['close'].rolling(7).mean()
        result['ma_30'] = result['close'].rolling(30).mean()
        result['ma_90'] = result['close'].rolling(90).mean()
        result['ma_200'] = result['close'].rolling(200).mean()

        # Price relative to moving averages
        result['price_vs_ma_30'] = (result['close'] - result['ma_30']) / result['ma_30']
        result['price_vs_ma_200'] = (result['close'] - result['ma_200']) / result['ma_200']

        # Golden cross / death cross
        result['ma_30_above_200'] = (result['ma_30'] > result['ma_200']).astype(int)

        # Volume momentum
        if 'volume' in result.columns:
            result['volume_ma_30'] = result['volume'].rolling(30).mean()
            result['volume_vs_ma'] = result['volume'] / result['volume_ma_30']

        print(f"[OK] Calculated {len(result.columns) - len(df.columns)} BTC momentum features")

        return result

    def get_btc_features(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        One-stop method to fetch BTC data and calculate all features.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price data and all momentum indicators
        """
        df = self.fetch_price_data(start_date, end_date)
        df = self.calculate_momentum_indicators(df)
        return df


def main():
    """Test the BTC data fetcher."""
    fetcher = BTCDataFetcher()

    # Fetch last 2 years of data
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    df = fetcher.get_btc_features(start_date=start_date)

    print("\nBTC data with momentum indicators:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")

    # Show latest values
    print("\nLatest BTC snapshot:")
    print(df.iloc[-1][['close', 'returns_30d', 'rsi', 'drawdown', 'volatility_30d']])

    # Check for strongly bullish/bearish signals
    latest = df.iloc[-1]
    print("\n--- Current BTC Momentum Assessment ---")
    print(f"Price: ${latest['close']:,.2f}")
    print(f"30-day return: {latest['returns_30d']*100:.2f}%")
    print(f"RSI: {latest['rsi']:.2f}")
    print(f"Drawdown from ATH: {latest['drawdown']*100:.2f}%")
    print(f"30-day volatility: {latest['volatility_30d']*100:.2f}%")

    if latest['rsi'] > 70:
        print("[!] RSI indicates OVERBOUGHT")
    elif latest['rsi'] < 30:
        print("[!] RSI indicates OVERSOLD")

    if latest['drawdown'] < -0.3:
        print("[!] Deep drawdown (>30%)")

    if latest['ma_30_above_200'] == 1:
        print("[OK] Bullish: 30-day MA above 200-day MA")
    else:
        print("[!] Bearish: 30-day MA below 200-day MA")


if __name__ == '__main__':
    main()
