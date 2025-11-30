"""
Wallet Performance Analytics

Comprehensive portfolio performance metrics for any wallet address.
Calculates institutional-grade metrics including:
- Sharpe Ratio, Sortino Ratio
- VaR/CVaR
- Beta to Bitcoin
- CAPM, Treynor Ratio
- CAGR, Cumulative Returns
- Maximum Drawdown
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime


class WalletPerformanceAnalyzer:
    """Analyze portfolio performance metrics for a wallet."""

    def __init__(self, portfolio_df: pd.DataFrame, risk_free_rate: float = 0.05):
        """
        Initialize with portfolio timeseries data.

        Args:
            portfolio_df: DataFrame with columns [date, symbol, balance, price_usd, value_usd]
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.portfolio_df = portfolio_df
        self.risk_free_rate = risk_free_rate
        self.daily_rf = (1 + risk_free_rate) ** (1/365) - 1

        # Build pivoted DataFrames
        self._build_timeseries()

    def _build_timeseries(self):
        """Build price, value, and weight timeseries."""
        df = self.portfolio_df.copy()

        # Pivot to get daily data per token
        self.prices = df.pivot(index="date", columns="symbol", values="price_usd").fillna(0)
        self.values = df.pivot(index="date", columns="symbol", values="value_usd").fillna(0)
        self.balances = df.pivot(index="date", columns="symbol", values="balance").fillna(0)

        # Total portfolio value
        self.total_value = self.values.sum(axis=1)
        self.total_value.name = "total_value"

        # Weights (composition)
        self.weights = self.values.div(self.total_value, axis=0).fillna(0)

        # Daily returns per token (log returns)
        self.token_returns = np.log(self.prices / self.prices.shift(1)).fillna(0)

        # Weighted portfolio returns
        self.portfolio_returns = (self.weights.shift(1) * self.token_returns).sum(axis=1)
        self.portfolio_returns.iloc[0] = 0  # First day has no return

        # Cumulative returns
        self.cumulative_returns = np.exp(self.portfolio_returns.cumsum()) - 1

    def calculate_sharpe_ratio(self, annualize: bool = True) -> float:
        """
        Calculate Sharpe Ratio.

        Sharpe = (Rp - Rf) / σp
        """
        excess_returns = self.portfolio_returns - self.daily_rf
        mean_excess = excess_returns.mean()
        std_returns = self.portfolio_returns.std()

        if std_returns == 0:
            return 0.0

        sharpe = mean_excess / std_returns

        if annualize:
            sharpe *= np.sqrt(365)

        return sharpe

    def calculate_sortino_ratio(self, annualize: bool = True) -> float:
        """
        Calculate Sortino Ratio (uses downside deviation instead of total std).

        Sortino = (Rp - Rf) / σd
        """
        excess_returns = self.portfolio_returns - self.daily_rf
        mean_excess = excess_returns.mean()

        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if mean_excess > 0 else 0.0

        downside_std = np.sqrt((downside_returns ** 2).mean())

        if downside_std == 0:
            return float('inf') if mean_excess > 0 else 0.0

        sortino = mean_excess / downside_std

        if annualize:
            sortino *= np.sqrt(365)

        return sortino

    def calculate_var_cvar(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall).

        Args:
            alpha: Confidence level (default 5% = 95% VaR)

        Returns:
            (VaR, CVaR) as percentages
        """
        returns = self.portfolio_returns.dropna()
        if len(returns) == 0:
            return 0.0, 0.0

        sorted_returns = np.sort(returns)
        index = int(alpha * len(sorted_returns))

        var = sorted_returns[index] if index < len(sorted_returns) else sorted_returns[-1]
        cvar = sorted_returns[:max(index, 1)].mean()

        return float(var), float(cvar)

    def calculate_max_drawdown(self) -> Tuple[float, datetime, datetime]:
        """
        Calculate Maximum Drawdown.

        Returns:
            (max_drawdown, peak_date, trough_date)
        """
        cumulative = self.cumulative_returns + 1  # Convert to wealth index
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        trough_idx = drawdown.idxmin()

        # Find peak before trough
        peak_idx = cumulative[:trough_idx].idxmax()

        return float(max_dd), peak_idx, trough_idx

    def calculate_beta(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate Beta relative to a benchmark (e.g., Bitcoin).

        Beta = Cov(Rp, Rb) / Var(Rb)

        Args:
            benchmark_returns: Series of benchmark daily returns with same index
        """
        # Normalize both indices to date only (strip timezone/time)
        port_ret = self.portfolio_returns.copy()
        port_ret.index = pd.to_datetime(port_ret.index).date

        bench_ret = benchmark_returns.copy()
        bench_ret.index = pd.to_datetime(bench_ret.index).date

        # Align indices
        common_idx = set(port_ret.index).intersection(set(bench_ret.index))
        if len(common_idx) < 2:
            return 0.0

        common_idx = sorted(list(common_idx))
        port_aligned = port_ret.loc[common_idx]
        bench_aligned = bench_ret.loc[common_idx]

        covariance = np.cov(port_aligned, bench_aligned)[0, 1]
        variance = bench_aligned.var()

        if variance == 0:
            return 0.0

        return covariance / variance

    def calculate_alpha(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate Jensen's Alpha (CAPM Alpha).

        Alpha = Rp - [Rf + Beta * (Rm - Rf)]

        Args:
            benchmark_returns: Series of benchmark daily returns
        """
        beta = self.calculate_beta(benchmark_returns)

        # Annualized returns (use log returns mean)
        port_annual = np.exp(self.portfolio_returns.mean() * 365) - 1
        bench_annual = np.exp(benchmark_returns.mean() * 365) - 1

        expected_return = self.risk_free_rate + beta * (bench_annual - self.risk_free_rate)
        alpha = port_annual - expected_return

        return alpha

    def calculate_treynor_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate Treynor Ratio.

        Treynor = (Rp - Rf) / Beta

        Args:
            benchmark_returns: Series of benchmark daily returns
        """
        beta = self.calculate_beta(benchmark_returns)
        if abs(beta) < 0.01:  # Avoid division by near-zero beta
            return 0.0

        # Annualized excess return
        port_annual = np.exp(self.portfolio_returns.mean() * 365) - 1
        excess_return = port_annual - self.risk_free_rate

        return excess_return / beta

    def calculate_cagr(self) -> float:
        """
        Calculate Compound Annual Growth Rate.

        CAGR = (End Value / Start Value) ^ (1 / years) - 1
        """
        start_value = self.total_value.iloc[0]
        end_value = self.total_value.iloc[-1]

        if start_value <= 0:
            return 0.0

        days = (self.total_value.index[-1] - self.total_value.index[0]).days
        years = days / 365.25 if days > 0 else 1

        cagr = (end_value / start_value) ** (1 / years) - 1
        return cagr

    def calculate_information_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate Information Ratio.

        IR = (Rp - Rb) / Tracking Error
        """
        # Normalize indices
        port_ret = self.portfolio_returns.copy()
        port_ret.index = pd.to_datetime(port_ret.index).date

        bench_ret = benchmark_returns.copy()
        bench_ret.index = pd.to_datetime(bench_ret.index).date

        common_idx = set(port_ret.index).intersection(set(bench_ret.index))
        if len(common_idx) < 2:
            return 0.0

        common_idx = sorted(list(common_idx))
        port_aligned = port_ret.loc[common_idx]
        bench_aligned = bench_ret.loc[common_idx]

        active_returns = port_aligned - bench_aligned
        tracking_error = active_returns.std()

        if tracking_error == 0:
            return 0.0

        ir = active_returns.mean() / tracking_error * np.sqrt(365)
        return ir

    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio.

        Calmar = CAGR / |Max Drawdown|
        """
        cagr = self.calculate_cagr()
        max_dd, _, _ = self.calculate_max_drawdown()

        if max_dd == 0:
            return float('inf') if cagr > 0 else 0.0

        return cagr / abs(max_dd)

    def get_all_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate all performance metrics.

        Args:
            benchmark_returns: Optional benchmark (e.g., BTC) returns for relative metrics

        Returns:
            Dictionary of all metrics
        """
        var, cvar = self.calculate_var_cvar()
        max_dd, peak_date, trough_date = self.calculate_max_drawdown()

        metrics = {
            # Return metrics
            "total_return": float(self.cumulative_returns.iloc[-1]),
            "cagr": self.calculate_cagr(),

            # Risk-adjusted metrics
            "sharpe_ratio": self.calculate_sharpe_ratio(),
            "sortino_ratio": self.calculate_sortino_ratio(),
            "calmar_ratio": self.calculate_calmar_ratio(),

            # Risk metrics
            "volatility": float(self.portfolio_returns.std() * np.sqrt(365)),
            "var_95": var,
            "cvar_95": cvar,
            "max_drawdown": max_dd,
            "max_drawdown_peak": str(peak_date),
            "max_drawdown_trough": str(trough_date),

            # Portfolio info
            "start_date": str(self.total_value.index[0]),
            "end_date": str(self.total_value.index[-1]),
            "start_value": float(self.total_value.iloc[0]),
            "end_value": float(self.total_value.iloc[-1]),
            "days": (self.total_value.index[-1] - self.total_value.index[0]).days,
        }

        # Relative metrics (if benchmark provided)
        if benchmark_returns is not None:
            metrics.update({
                "beta": self.calculate_beta(benchmark_returns),
                "alpha": self.calculate_alpha(benchmark_returns),
                "treynor_ratio": self.calculate_treynor_ratio(benchmark_returns),
                "information_ratio": self.calculate_information_ratio(benchmark_returns),
            })

        return metrics

    def get_composition_timeseries(self) -> pd.DataFrame:
        """Get portfolio composition (weights) over time."""
        return self.weights

    def get_returns_timeseries(self) -> pd.DataFrame:
        """Get daily and cumulative returns over time."""
        return pd.DataFrame({
            "daily_return": self.portfolio_returns,
            "cumulative_return": self.cumulative_returns,
            "total_value": self.total_value
        })

    def normalize_returns(self, start_value: float = 10000) -> pd.Series:
        """
        Normalize returns to a starting value for comparison.

        Args:
            start_value: Initial portfolio value (default $10,000)
        """
        return start_value * (1 + self.cumulative_returns)

    def calculate_rolling_sharpe(self, window: int = 30) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            window: Rolling window in days
        """
        rolling_mean = self.portfolio_returns.rolling(window).mean()
        rolling_std = self.portfolio_returns.rolling(window).std()

        # Annualized rolling Sharpe
        rolling_sharpe = ((rolling_mean - self.daily_rf) / rolling_std) * np.sqrt(365)
        return rolling_sharpe.replace([np.inf, -np.inf], np.nan)

    def calculate_rolling_sortino(self, window: int = 30) -> pd.Series:
        """
        Calculate rolling Sortino ratio.

        Args:
            window: Rolling window in days
        """
        def sortino_window(returns):
            if len(returns) < window:
                return np.nan
            excess = returns - self.daily_rf
            mean_excess = excess.mean()
            downside = excess[excess < 0]
            if len(downside) == 0:
                return np.nan
            downside_std = np.sqrt((downside ** 2).mean())
            if downside_std == 0:
                return np.nan
            return (mean_excess / downside_std) * np.sqrt(365)

        rolling_sortino = self.portfolio_returns.rolling(window).apply(
            sortino_window, raw=False
        )
        return rolling_sortino.replace([np.inf, -np.inf], np.nan)

    def calculate_rolling_volatility(self, window: int = 30) -> pd.Series:
        """
        Calculate rolling annualized volatility.

        Args:
            window: Rolling window in days
        """
        rolling_vol = self.portfolio_returns.rolling(window).std() * np.sqrt(365)
        return rolling_vol

    def calculate_rolling_beta(self, benchmark_returns: pd.Series, window: int = 30) -> pd.Series:
        """
        Calculate rolling beta relative to benchmark.

        Args:
            benchmark_returns: Series of benchmark daily returns
            window: Rolling window in days
        """
        # Normalize indices
        port_ret = self.portfolio_returns.copy()
        port_ret.index = pd.to_datetime(port_ret.index).date

        bench_ret = benchmark_returns.copy()
        bench_ret.index = pd.to_datetime(bench_ret.index).date

        # Align to common dates
        common_idx = sorted(set(port_ret.index).intersection(set(bench_ret.index)))
        if len(common_idx) < window:
            return pd.Series(index=common_idx, dtype=float)

        port_aligned = port_ret.loc[common_idx]
        bench_aligned = bench_ret.loc[common_idx]

        # Create aligned DataFrame
        aligned_df = pd.DataFrame({
            'portfolio': port_aligned.values,
            'benchmark': bench_aligned.values
        }, index=common_idx)

        def calc_beta(df):
            if len(df) < window:
                return np.nan
            cov = np.cov(df['portfolio'], df['benchmark'])[0, 1]
            var = df['benchmark'].var()
            if var == 0:
                return np.nan
            return cov / var

        rolling_beta = aligned_df.rolling(window).apply(
            lambda x: calc_beta(pd.DataFrame({'portfolio': x.iloc[:, 0], 'benchmark': x.iloc[:, 1]})),
            raw=False
        )

        # Use portfolio column result
        return pd.Series(
            rolling_beta['portfolio'].values,
            index=rolling_beta.index
        ).replace([np.inf, -np.inf], np.nan)

    def get_rolling_metrics(
        self,
        window: int = 30,
        metrics: list = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Get multiple rolling metrics.

        Args:
            window: Rolling window in days
            metrics: List of metrics to calculate ['sharpe', 'sortino', 'volatility', 'beta']
            benchmark_returns: Required for beta calculation

        Returns:
            Dict with dates and rolling metric arrays
        """
        if metrics is None:
            metrics = ['sharpe', 'sortino', 'volatility']

        result = {
            'dates': [str(d) for d in self.portfolio_returns.index],
        }

        if 'sharpe' in metrics:
            rolling_sharpe = self.calculate_rolling_sharpe(window)
            result['rolling_sharpe'] = [
                None if pd.isna(v) else round(float(v), 4)
                for v in rolling_sharpe.values
            ]

        if 'sortino' in metrics:
            rolling_sortino = self.calculate_rolling_sortino(window)
            result['rolling_sortino'] = [
                None if pd.isna(v) else round(float(v), 4)
                for v in rolling_sortino.values
            ]

        if 'volatility' in metrics:
            rolling_vol = self.calculate_rolling_volatility(window)
            result['rolling_volatility'] = [
                None if pd.isna(v) else round(float(v), 4)
                for v in rolling_vol.values
            ]

        if 'beta' in metrics and benchmark_returns is not None:
            rolling_beta = self.calculate_rolling_beta(benchmark_returns, window)
            # Align to main dates
            beta_dict = dict(zip(rolling_beta.index, rolling_beta.values))
            result['rolling_beta'] = [
                None if pd.isna(beta_dict.get(d)) else round(float(beta_dict.get(d, np.nan)), 4)
                for d in self.portfolio_returns.index
            ]

        return result
