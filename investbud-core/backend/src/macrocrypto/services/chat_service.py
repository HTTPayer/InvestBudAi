"""
Chat service for managing conversational AI with portfolio context.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from ..db.models import ChatSession
from ..utils import get_llm_client


class ChatService:
    """Service for managing chat sessions with LLM."""

    def __init__(self, db_session: Session, wallet_analyzer=None):
        """
        Initialize chat service.

        Args:
            db_session: SQLAlchemy database session
            wallet_analyzer: Optional WalletAnalyzer instance for portfolio data
        """
        self.db = db_session
        self.wallet_analyzer = wallet_analyzer
        self.max_history = 20  # Keep last 20 messages
        self.portfolio_cache_ttl = timedelta(minutes=5)  # Cache portfolio for 5 minutes

    async def chat(
        self,
        session_id: str,
        message: str,
        wallet_address: Optional[str] = None,
        network: Optional[str] = None,
        regime_context: Optional[Dict] = None,
        news_context: Optional[Dict] = None,
        wallet_performance_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message with context.

        Args:
            session_id: Unique session identifier
            message: User message
            wallet_address: Optional wallet address for portfolio context
            network: Optional network (eth-mainnet, arbitrum-mainnet, etc.)
            regime_context: Optional pre-fetched regime data
            news_context: Optional cached news/smart money analysis
            wallet_performance_context: Optional wallet performance metrics

        Returns:
            Dictionary with response and metadata
        """
        # Get or create session
        session = self.get_or_create_session(session_id, wallet_address, network)

        # Use stored wallet_address/network from session if not provided in request
        effective_wallet = wallet_address or session.wallet_address
        effective_network = network or session.network or 'eth-mainnet'

        # Get conversation history
        history = session.get_messages()

        # Fetch or get cached portfolio data
        portfolio_data = None
        if effective_wallet and self.wallet_analyzer:
            portfolio_data = await self._get_portfolio_data(session, effective_wallet, effective_network)

        # Build system context
        context = await self._build_context(
            wallet_address=effective_wallet,
            regime_context=regime_context,
            portfolio_data=portfolio_data,
            news_context=news_context,
            wallet_performance_context=wallet_performance_context
        )

        # Call LLM
        llm_client = get_llm_client()
        response = await self._call_llm(message, history, context, llm_client)

        # Update history
        history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Keep only last N messages
        history = history[-self.max_history:]

        # Save to database
        session.set_messages(history)
        session.updated_at = datetime.utcnow()
        self.db.commit()

        return {
            "session_id": session_id,
            "response": response,
            "message_count": len(history),
            "context_used": {
                "wallet": effective_wallet is not None,
                "portfolio": portfolio_data is not None,
                "regime": regime_context is not None,
                "news": news_context is not None,
                "performance": wallet_performance_context is not None
            }
        }

    def get_or_create_session(
        self,
        session_id: str,
        wallet_address: Optional[str] = None,
        network: Optional[str] = None
    ) -> ChatSession:
        """
        Get existing session or create new one.

        Args:
            session_id: Session identifier
            wallet_address: Optional wallet address
            network: Optional network

        Returns:
            ChatSession object
        """
        session = self.db.query(ChatSession).filter_by(session_id=session_id).first()

        if not session:
            session = ChatSession(
                session_id=session_id,
                wallet_address=wallet_address,
                network=network or 'eth-mainnet',
                messages='[]'
            )
            self.db.add(session)
            self.db.commit()
        else:
            # Update wallet/network if provided and changed
            if wallet_address and session.wallet_address != wallet_address:
                session.wallet_address = wallet_address
                session.portfolio_snapshot = None  # Invalidate cache
            if network and session.network != network:
                session.network = network
                session.portfolio_snapshot = None  # Invalidate cache

        return session

    async def _get_portfolio_data(
        self,
        session: ChatSession,
        wallet_address: str,
        network: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get portfolio data from cache or fetch fresh.

        Args:
            session: ChatSession object
            wallet_address: Wallet address
            network: Network to query

        Returns:
            Portfolio data dictionary or None
        """
        network = network or session.network or 'eth-mainnet'

        # Check if cache is valid
        if session.portfolio_snapshot and session.portfolio_updated_at:
            cache_age = datetime.utcnow() - session.portfolio_updated_at
            if cache_age < self.portfolio_cache_ttl:
                print(f"[OK] Using cached portfolio data (age: {cache_age.seconds}s)")
                return session.get_portfolio_snapshot()

        # Fetch fresh data
        if not self.wallet_analyzer:
            return None

        try:
            print(f"[OK] Fetching fresh portfolio data for {wallet_address}")
            analysis = self.wallet_analyzer.analyze_wallet(wallet_address, network)

            if 'error' in analysis:
                print(f"[!] Portfolio fetch failed: {analysis['error']}")
                return None

            # Cache the results
            session.set_portfolio_snapshot(analysis)
            self.db.commit()

            return analysis

        except Exception as e:
            print(f"[!] Portfolio fetch error: {e}")
            return None

    async def _build_context(
        self,
        wallet_address: Optional[str],
        regime_context: Optional[Dict],
        portfolio_data: Optional[Dict] = None,
        news_context: Optional[Dict] = None,
        wallet_performance_context: Optional[Dict] = None
    ) -> str:
        """
        Build context string for LLM.

        Args:
            wallet_address: Optional wallet address
            regime_context: Optional regime data
            portfolio_data: Optional portfolio analysis data
            news_context: Optional news/smart money analysis
            wallet_performance_context: Optional wallet performance metrics

        Returns:
            Context string
        """
        context_parts = []

        # Add regime context
        if regime_context:
            context_parts.append(f"""
Current Macro Regime: {regime_context.get('regime', 'Unknown')}
Confidence: {regime_context.get('confidence', 0) * 100:.0f}%
Risk-On Probability: {regime_context.get('risk_on_probability', 0) * 100:.0f}%

Key Indicators:
- BTC Price: ${regime_context.get('features', {}).get('btc_price', 'N/A'):,.2f}
- BTC 30D Returns: {regime_context.get('features', {}).get('btc_returns_30d', 0) * 100:.1f}%
- VIX: {regime_context.get('features', {}).get('vix', 'N/A')}
- Fed Funds Rate: {regime_context.get('features', {}).get('fed_funds', 'N/A')}%
""")

        # Add portfolio context
        if portfolio_data:
            total_value = portfolio_data.get('total_value_usd', 0)
            risk_alloc = portfolio_data.get('risk_allocation', 0) * 100
            stable_alloc = portfolio_data.get('stable_allocation', 0) * 100

            context_parts.append(f"""
User Portfolio:
- Wallet: {wallet_address}
- Total Value: ${total_value:,.2f}
- Risk Allocation: {risk_alloc:.1f}%
- Stable Allocation: {stable_alloc:.1f}%
""")

            # Add top holdings
            holdings = portfolio_data.get('portfolio', [])
            if holdings:
                context_parts.append("Top Holdings:")
                for holding in holdings[:5]:  # Top 5 only
                    symbol = holding.get('symbol', 'UNKNOWN')
                    weight = holding.get('weight', 0) * 100
                    value = holding.get('value_usd', 0)
                    context_parts.append(f"  - {symbol}: {weight:.1f}% (${value:,.2f})")
        elif wallet_address:
            context_parts.append(f"User Wallet: {wallet_address} (portfolio data not available)")

        # Add news/smart money context
        if news_context:
            analysis = news_context.get('analysis', '')
            generated_at = news_context.get('generated_at', 'Unknown')
            # Truncate if too long to avoid context overflow
            if len(analysis) > 2000:
                analysis = analysis[:2000] + "..."
            context_parts.append(f"""
Recent Smart Money Analysis (as of {generated_at}):
{analysis}
""")

        # Add wallet performance metrics
        if wallet_performance_context:
            total_return = wallet_performance_context.get('total_return', 0) * 100
            sharpe = wallet_performance_context.get('sharpe_ratio', 0)
            sortino = wallet_performance_context.get('sortino_ratio', 0)
            max_dd = wallet_performance_context.get('max_drawdown', 0) * 100
            volatility = wallet_performance_context.get('volatility', 0) * 100
            beta = wallet_performance_context.get('beta')
            var_95 = wallet_performance_context.get('var_95', 0) * 100
            days = wallet_performance_context.get('days', 0)

            perf_context = f"""
Portfolio Performance Metrics ({days} days):
- Total Return: {total_return:.1f}%
- Sharpe Ratio: {sharpe:.2f}
- Sortino Ratio: {sortino:.2f}
- Max Drawdown: {max_dd:.1f}%
- Volatility (Ann): {volatility:.1f}%
- VaR (95%): {var_95:.1f}%"""

            if beta is not None:
                alpha = wallet_performance_context.get('alpha', 0) * 100
                perf_context += f"""
- Beta vs BTC: {beta:.2f}
- Alpha (Ann): {alpha:.1f}%"""

            context_parts.append(perf_context)

        if not context_parts:
            return "No additional context available."

        return "\n".join(context_parts)

    async def _call_llm(
        self,
        message: str,
        history: List[Dict],
        context: str,
        llm_client
    ) -> str:
        """
        Call LLM with conversation history and context.

        Args:
            message: Current user message
            history: Conversation history
            context: System context
            llm_client: LLM client instance

        Returns:
            LLM response
        """
        # Build messages array
        messages = [
            {
                "role": "system",
                "content": f"""You are a professional crypto investment advisor with expertise in macro-economic analysis.

You help users understand:
- Current macro regime (Risk-On vs Risk-Off)
- Portfolio allocation strategies
- Market conditions and their implications

Guidelines:
- Be concise but informative
- Use data from the context when available
- Provide actionable insights
- Maintain a professional tone
- Don't use emojis

Context:
{context}"""
            }
        ]

        # Add conversation history (user/assistant pairs only)
        for msg in history:
            if msg.get('role') in ['user', 'assistant']:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })

        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })

        # Call LLM API
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    llm_client.api_url,
                    json={
                        "messages": messages,
                        "model": llm_client.model,
                        "temperature": llm_client.temperature
                    },
                    headers={
                        "x-api-key": llm_client.api_key,
                        "Content-Type": "application/json"
                    }
                )
                response.raise_for_status()
                result = response.json()

                # Extract response
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                elif "message" in result:
                    return result["message"]
                else:
                    return result.get("content", str(result))

        except Exception as e:
            print(f"[ERROR] LLM API call failed: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."

    def get_session_history(self, session_id: str) -> List[Dict]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of messages
        """
        session = self.db.query(ChatSession).filter_by(session_id=session_id).first()
        return session.get_messages() if session else []

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        session = self.db.query(ChatSession).filter_by(session_id=session_id).first()
        if session:
            self.db.delete(session)
            self.db.commit()
            return True
        return False

    def cleanup_old_sessions(self, days: int = 7) -> int:
        """
        Clean up sessions older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of sessions deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        old_sessions = self.db.query(ChatSession).filter(
            ChatSession.updated_at < cutoff
        ).all()

        count = len(old_sessions)
        for session in old_sessions:
            self.db.delete(session)

        self.db.commit()
        return count
