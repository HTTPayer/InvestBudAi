"""
LLM client for generating natural language investment advice.
Uses httpayer LLM API.
"""
import os
import httpx
from typing import Dict, Any, List


class LLMAdvisoryClient:
    """Client for generating investment advice using LLM."""

    def __init__(self, api_key: str = None):
        """
        Initialize LLM client.

        Args:
            api_key: API key for httpayer. If None, reads from SERVER_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("SERVER_API_KEY")
        if not self.api_key:
            raise ValueError("SERVER_API_KEY not found in environment variables")

        self.api_url = "https://api.httpayer.com/llm/chat"
        self.model = "gpt-4o-mini"
        self.temperature = 0.7

    async def generate_recommendation(
        self,
        wallet_data: Dict[str, Any],
        regime_data: Dict[str, Any],
        quantitative_signal: Dict[str, Any],
        performance_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate LLM-powered recommendation and regime explanation.

        Args:
            wallet_data: Dictionary with wallet analysis (total_value, portfolio, allocations)
            regime_data: Dictionary with regime prediction (regime, confidence, features)
            quantitative_signal: Dictionary with quantitative recommendation (action, optimal allocation)
            performance_data: Optional dictionary with historical performance metrics (Sharpe, VaR, etc.)

        Returns:
            Dictionary with:
                - explanation: Brief explanation of the regime
                - summary: Brief recommendation summary
                - actionable_steps: List of 3-4 specific action items
        """
        # Build context for the LLM
        portfolio_summary = self._format_portfolio(wallet_data)
        regime_summary = self._format_regime(regime_data)
        performance_summary = self._format_performance(performance_data) if performance_data else None

        # Create prompt
        system_prompt = """You are a professional crypto investment advisor with expertise in macro-economic analysis and portfolio risk management.

Your role is to analyze market conditions, portfolio data, and historical performance to provide:
1. A brief explanation of the current macro regime (1-2 sentences)
2. A concise recommendation summary (1 sentence)
3. A list of 3-4 specific, actionable steps

Format your response as JSON:
{
  "explanation": "Brief regime explanation...",
  "summary": "One sentence recommendation...",
  "actionable_steps": [
    "Specific action 1",
    "Specific action 2",
    "Specific action 3"
  ]
}

Guidelines:
- Be specific with percentages and token names
- Focus on actionable steps (e.g., "Swap 30% ETH to USDC")
- Consider historical performance metrics (Sharpe ratio, drawdowns, volatility) in your recommendations
- Keep explanations concise and professional
- Do not use emojis or overly casual language"""

        user_prompt = f"""Provide investment advice based on this analysis:

WALLET ANALYSIS:
{portfolio_summary}

MACRO REGIME:
{regime_summary}
{f'''
HISTORICAL PERFORMANCE:
{performance_summary}''' if performance_summary else ''}

QUANTITATIVE SIGNAL:
Action: {quantitative_signal.get('action', 'MAINTAIN')}
Current Risk Allocation: {quantitative_signal.get('current_risk_allocation', 0):.1%}
Optimal Risk Allocation: {quantitative_signal.get('optimal_risk_allocation', 0.5):.1%}"""

        # Make API call
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    json={
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        "model": self.model,
                        "temperature": self.temperature
                    },
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                result = response.json()

                # Extract message content from response
                content = None
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                elif "message" in result:
                    content = result["message"]
                else:
                    content = result.get("content", str(result))

                # Parse JSON response
                import json
                import re

                # Try to extract JSON from markdown code blocks if present
                json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content)
                if json_match:
                    content = json_match.group(1)

                try:
                    parsed = json.loads(content)
                    return {
                        "explanation": parsed.get("explanation", ""),
                        "summary": parsed.get("summary", ""),
                        "actionable_steps": parsed.get("actionable_steps", [])
                    }
                except json.JSONDecodeError:
                    # Fallback if JSON parsing fails
                    return self._fallback_recommendation(quantitative_signal)

        except httpx.HTTPError as e:
            print(f"[!] LLM API error: {str(e)}")
            return self._fallback_recommendation(quantitative_signal)
        except Exception as e:
            print(f"[!] Unexpected error in LLM client: {str(e)}")
            return self._fallback_recommendation(quantitative_signal)

    def _format_portfolio(self, wallet_data: Dict[str, Any]) -> str:
        """Format portfolio data for LLM prompt."""
        total_value = wallet_data.get('total_value_usd', 0)
        risk_alloc = wallet_data.get('risk_allocation', 0)
        stable_alloc = wallet_data.get('stable_allocation', 0)

        portfolio_items = []
        for holding in wallet_data.get('portfolio', [])[:5]:  # Top 5 holdings
            symbol = holding.get('symbol', 'UNKNOWN')
            value = holding.get('value_usd', 0)
            weight = holding.get('weight', 0)
            portfolio_items.append(f"  - {symbol}: ${value:,.2f} ({weight:.1%})")

        return f"""Total Portfolio Value: ${total_value:,.2f}
Risk Allocation (Crypto): {risk_alloc:.1%}
Stable Allocation (Stablecoins): {stable_alloc:.1%}

Top Holdings:
{chr(10).join(portfolio_items)}"""

    def _format_regime(self, regime_data: Dict[str, Any]) -> str:
        """Format regime data for LLM prompt."""
        regime = regime_data.get('regime', 'Unknown')
        confidence = regime_data.get('confidence', 0)
        risk_on_prob = regime_data.get('risk_on_probability', 0)

        features = regime_data.get('features', {})
        btc_price = features.get('btc_price', 0)
        btc_returns = features.get('btc_returns_30d', 0)
        vix = features.get('vix', 0)
        fed_funds = features.get('fed_funds', 0)

        return f"""Current Regime: {regime}
Confidence: {confidence:.1%}
Risk-On Probability: {risk_on_prob:.1%}

Key Indicators:
  - BTC Price: ${btc_price:,.2f}
  - BTC 30D Returns: {btc_returns:.1%}
  - VIX: {vix:.2f}
  - Fed Funds Rate: {fed_funds:.2f}%"""

    def _format_performance(self, performance_data: Dict[str, Any]) -> str:
        """Format historical performance data for LLM prompt."""
        if not performance_data:
            return "No historical performance data available."

        total_return = performance_data.get('total_return', 0)
        cagr = performance_data.get('cagr', 0)
        sharpe = performance_data.get('sharpe_ratio', 0)
        sortino = performance_data.get('sortino_ratio', 0)
        volatility = performance_data.get('volatility', 0)
        max_drawdown = performance_data.get('max_drawdown', 0)
        var_95 = performance_data.get('var_95', 0)
        cvar_95 = performance_data.get('cvar_95', 0)
        beta = performance_data.get('beta')
        alpha = performance_data.get('alpha')
        days = performance_data.get('days', 0)
        start_date = performance_data.get('start_date', 'N/A')
        end_date = performance_data.get('end_date', 'N/A')

        lines = [
            f"Period: {start_date} to {end_date} ({days} days)",
            f"Total Return: {total_return:.1%}",
            f"CAGR: {cagr:.1%}",
            f"Sharpe Ratio: {sharpe:.2f}",
            f"Sortino Ratio: {sortino:.2f}",
            f"Volatility (Annualized): {volatility:.1%}",
            f"Max Drawdown: {max_drawdown:.1%}",
            f"VaR (95%): {var_95:.1%}",
            f"CVaR (95%): {cvar_95:.1%}",
        ]

        if beta is not None:
            lines.append(f"Beta (vs BTC): {beta:.2f}")
        if alpha is not None:
            lines.append(f"Alpha (Annualized): {alpha:.1%}")

        return "\n".join(lines)

    def _fallback_recommendation(self, quantitative_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback recommendation if LLM API fails."""
        action = quantitative_signal.get('action', 'MAINTAIN')
        current_alloc = quantitative_signal.get('current_risk_allocation', 0)
        optimal_alloc = quantitative_signal.get('optimal_risk_allocation', 0.5)

        regime = quantitative_signal.get('regime', 'Unknown')

        explanation = f"Current market regime is {regime}. Our model suggests adjusting portfolio allocation based on macro indicators."

        if action == 'INCREASE':
            summary = f"Increase crypto exposure from {current_alloc:.0%} to {optimal_alloc:.0%}"
            steps = [
                f"Rotate stablecoins into crypto assets",
                f"Target {optimal_alloc:.0%} allocation in BTC/ETH",
                "Monitor regime for potential reversal"
            ]
        elif action == 'DECREASE':
            summary = f"Reduce crypto exposure from {current_alloc:.0%} to {optimal_alloc:.0%}"
            steps = [
                f"Swap crypto assets to stablecoins",
                f"Target {optimal_alloc:.0%} allocation to reduce risk",
                "Wait for Risk-On signals before re-entering"
            ]
        else:
            summary = f"Maintain current {current_alloc:.0%} crypto allocation"
            steps = [
                "Keep current portfolio composition",
                "Monitor macro regime changes",
                "Be ready to rebalance if regime shifts"
            ]

        return {
            "explanation": explanation,
            "summary": summary,
            "actionable_steps": steps
        }


    async def generate_advisory(
        self,
        wallet_data: Dict[str, Any],
        regime_data: Dict[str, Any],
        recommendation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Wrapper for backwards compatibility with standalone /advisory endpoint.
        Calls generate_recommendation internally.
        """
        return await self.generate_recommendation(
            wallet_data=wallet_data,
            regime_data=regime_data,
            quantitative_signal=recommendation
        )


# Singleton instance
_llm_client = None


def get_llm_client() -> LLMAdvisoryClient:
    """Get or create the LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMAdvisoryClient()
    return _llm_client
