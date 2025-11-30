"""Utility functions and performance metrics."""
from .metrics import (
    calculate_returns,
    calculate_cagr,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_all_metrics,
    print_metrics_report,
    compare_metrics
)
from .llm_client import LLMAdvisoryClient, get_llm_client
from .advanced_metrics import (
    calculate_beta,
    calculate_excess_return,
    calculate_treynor_ratio,
    calculate_alpha,
    calculate_all_advanced_metrics,
    calculate_portfolio_vs_btc_metrics
)

__all__ = [
    'calculate_returns',
    'calculate_cagr',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_all_metrics',
    'print_metrics_report',
    'compare_metrics',
    'LLMAdvisoryClient',
    'get_llm_client',
    'calculate_beta',
    'calculate_excess_return',
    'calculate_treynor_ratio',
    'calculate_alpha',
    'calculate_all_advanced_metrics',
    'calculate_portfolio_vs_btc_metrics'
]
