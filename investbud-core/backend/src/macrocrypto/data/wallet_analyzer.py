"""
Wallet analyzer: Fetch on-chain holdings and calculate portfolio metrics.
"""
import os
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import requests
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv

from ..utils.metrics import calculate_all_metrics

if TYPE_CHECKING:
    from ..models import MacroRegimeClassifier

load_dotenv()

def coingecko_api(url, coins=None, use_cached_data=False, max_retries=5):
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'original')
    os.makedirs(data_dir, exist_ok=True)

    if use_cached_data:
        if url == 'FDV':
            return pd.read_csv(os.path.join(data_dir, f'{coins}_fdv_data.csv'))
        elif url == 'LIST':
            return pd.read_csv(os.path.join(data_dir, f'{coins}_list_data.csv'))
        elif url == 'PLATFORM':
            return pd.read_csv(os.path.join(data_dir, f'{coins}_platform_data.csv'))

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": os.getenv('COINGECKO_API_KEY', '')

    }

    def make_request(request_url, retries=max_retries):
        delay = 2  # Start with a 2-second delay

        for attempt in range(retries):
            response = requests.get(request_url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                print(f"Rate limit hit. Retrying after {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return None

        print("Max retries exceeded.")
        return None

    # Handle FDV case with pagination
    if url == 'FDV':
        if coins is None:
            base_url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&per_page=250"
        elif isinstance(coins, list):
            coin_ids = ",".join(coins)
            base_url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={coin_ids}&per_page=250"
        else:
            base_url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&category={coins}&per_page=250"

        all_data = []
        page = 1

        while True:
            paginated_url = f"{base_url}&page={page}"
            data = make_request(paginated_url)

            if not data:
                break

            all_data.extend(data)
            print(f"Retrieved page {page} with {len(data)} records.")
            page += 1

            # To avoid hitting rate limits, sleep for a short period between requests
            time.sleep(2)

        df = pd.DataFrame(all_data)
        df.to_csv(os.path.join(data_dir, f'{coins}_fdv_data.csv'), index=False)

    elif url == 'LIST':
        base_url = "https://api.coingecko.com/api/v3/coins/list?include_platform=true"
        data = make_request(base_url)
        if data:
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(data_dir, f'{coins}_list_data.csv'), index=False)

    elif url == 'PLATFORM':
        base_url = "https://api.coingecko.com/api/v3/asset_platforms"
        data = make_request(base_url)
        if data:
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(data_dir, f'{coins}_platform_data.csv'), index=False)

    else:
        data = make_request(url)
        if data:
            df = pd.DataFrame(data)

    return df


class WalletAnalyzer:
    """
    Analyze crypto wallets by fetching on-chain holdings and calculating metrics.

    Supports:
    - Ethereum (and EVM-compatible chains)
    - Token balances and prices via Alchemy Portfolio API
    - Fallback price data via CoinGecko API (if needed)
    - Portfolio metrics calculation
    - Regime-based recommendations
    """

    # Common token contract addresses (Ethereum mainnet)
    KNOWN_TOKENS = {
        'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
        'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
        'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
        'WBTC': '0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599',
        'UNI': '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
        'LINK': '0x514910771AF9Ca656af840dff83E8264EcF986CA',
        'AAVE': '0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9',
    }

    # CoinGecko token IDs for price fetching
    COINGECKO_IDS = {
        'ETH': 'ethereum',
        'WETH': 'weth',
        'WBTC': 'wrapped-bitcoin',
        'USDC': 'usd-coin',
        'USDT': 'tether',
        'DAI': 'dai',
        'UNI': 'uniswap',
        'LINK': 'chainlink',
        'AAVE': 'aave',
        'BTC': 'bitcoin'
    }

    def __init__(
        self,
        alchemy_api_key: Optional[str] = None,
        coingecko_api_key: Optional[str] = None
    ):
        """
        Initialize wallet analyzer.

        Args:
            alchemy_api_key: Alchemy API key (reads from ALCHEMY_API_KEY env var if not provided)
            coingecko_api_key: CoinGecko API key (reads from COINGECKO_API_KEY env var if not provided)
        """
        self.alchemy_api_key = alchemy_api_key or os.getenv('ALCHEMY_API_KEY')
        self.coingecko_api_key = coingecko_api_key or os.getenv('COINGECKO_API_KEY')

        if not self.alchemy_api_key:
            print("[WARNING] No Alchemy API key found. Set ALCHEMY_API_KEY environment variable.")

    def fetch_token_balances(
        self,
        wallet_address: str,
        network: str = 'eth-mainnet'
    ) -> List[Dict]:
        """
        Fetch ERC-20 token balances for a wallet using Alchemy Portfolio API.

        Args:
            wallet_address: Ethereum wallet address
            network: Network to query (default: eth-mainnet)

        Returns:
            List of token balance dictionaries
        """
        if not self.alchemy_api_key:
            raise ValueError("Alchemy API key required. Set ALCHEMY_API_KEY environment variable.")

        # Use Alchemy Portfolio API
        url = f"https://api.g.alchemy.com/data/v1/{self.alchemy_api_key}/assets/tokens/by-address"

        payload = {
            "addresses": [
                {
                    "address": wallet_address,
                    "networks": [network]
                }
            ]
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] Alchemy API error: {e}")
            print(f"Response: {response.text}")
            raise ValueError(f"Failed to fetch token balances: {e}")

        balances = []

        # Parse response
        if 'data' in data and 'tokens' in data['data']:
            tokens = data['data']['tokens']
            # print(f'tokens: {tokens}')

            for token in tokens:
                # Skip if error
                if token.get('error'):
                    continue

                # Get contract address and balance first
                contract_address = token.get('tokenAddress')
                balance_raw = token.get('tokenBalance', '0')

                # Special handling for native ETH (tokenAddress is None)
                if contract_address is None:
                    symbol = 'ETH'
                    name = 'Ethereum'
                    decimals = 18
                else:
                    # Get token metadata
                    metadata = token.get('tokenMetadata', {})
                    symbol = metadata.get('symbol') or 'UNKNOWN'
                    name = metadata.get('name') or 'Unknown Token'
                    decimals = metadata.get('decimals')
                    if decimals is None:
                        decimals = 18

                # Convert balance from string (in wei) to float
                try:
                    if balance_raw and balance_raw != '0' and balance_raw != '0x0':
                        # Handle hex string
                        if isinstance(balance_raw, str) and balance_raw.startswith('0x'):
                            balance = int(balance_raw, 16) / (10 ** decimals)
                        else:
                            balance = int(balance_raw) / (10 ** decimals)
                    else:
                        balance = 0
                except (ValueError, TypeError) as e:
                    print(f"[WARNING] Failed to parse balance for {symbol}: {e}")
                    balance = 0

                if balance > 0:
                    # Get price if available
                    price_usd = None
                    token_prices = token.get('tokenPrices', [])
                    if token_prices and len(token_prices) > 0:
                        for price_data in token_prices:
                            if price_data.get('currency') == 'usd':
                                try:
                                    price_usd = float(price_data.get('value', 0))
                                except (ValueError, TypeError):
                                    price_usd = None
                                break

                    balances.append({
                        'symbol': symbol,
                        'name': name,
                        'balance': balance,
                        'contract_address': contract_address,
                        'decimals': decimals,
                        'price_usd': price_usd
                    })

        print(f"[OK] Found {len(balances)} tokens in wallet")
        return balances

    def fetch_token_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch current USD prices for tokens using CoinGecko API.

        Args:
            symbols: List of token symbols (e.g., ['ETH', 'WBTC', 'USDC'])

        Returns:
            Dictionary of {symbol: price_usd}
        """
        # Map symbols to CoinGecko IDs
        coingecko_ids = []
        symbol_to_id = {}

        for symbol in symbols:
            cg_id = self.COINGECKO_IDS.get(symbol)
            if cg_id:
                coingecko_ids.append(cg_id)
                symbol_to_id[symbol] = cg_id
            else:
                print(f"[WARNING] No CoinGecko ID found for {symbol}")

        if not coingecko_ids:
            return {}

        # Fetch prices from CoinGecko
        ids_str = ','.join(coingecko_ids)
        url = f"https://api.coingecko.com/api/v3/simple/price"

        params = {
            'ids': ids_str,
            'vs_currencies': 'usd'
        }

        headers = {}
        if self.coingecko_api_key:
            headers['x-cg-demo-api-key'] = self.coingecko_api_key

        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        # Map back to symbols
        prices = {}
        for symbol, cg_id in symbol_to_id.items():
            if cg_id in data and 'usd' in data[cg_id]:
                prices[symbol] = data[cg_id]['usd']

        print(f"[OK] Fetched prices for {len(prices)} tokens")
        return prices

    def calculate_portfolio_composition(
        self,
        balances: List[Dict],
        prices: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Calculate portfolio composition with USD values and weights.

        Args:
            balances: List of token balances (with optional price_usd field)
            prices: Optional dictionary of token prices (fallback if not in balances)

        Returns:
            DataFrame with portfolio composition
        """
        portfolio = []

        for token in balances:
            symbol = token['symbol']
            balance = token['balance']

            # Use price from Alchemy if available, otherwise from prices dict
            price = token.get('price_usd')
            if price is None and prices:
                price = prices.get(symbol, 0)
            if price is None:
                price = 0

            value_usd = balance * price

            portfolio.append({
                'symbol': symbol,
                'name': token['name'],
                'balance': balance,
                'price_usd': price,
                'value_usd': value_usd
            })

        df = pd.DataFrame(portfolio)

        if not df.empty:
            total_value = df['value_usd'].sum()
            df['weight'] = df['value_usd'] / total_value
            df = df.sort_values('value_usd', ascending=False)

        return df

    def analyze_wallet(
        self,
        wallet_address: str,
        network: str = 'eth-mainnet'
    ) -> Dict:
        """
        Complete wallet analysis: fetch holdings, get prices, calculate metrics.

        Args:
            wallet_address: Ethereum wallet address
            network: Network to query

        Returns:
            Dictionary with portfolio analysis
        """
        print("\n" + "=" * 70)
        print(f"ANALYZING WALLET: {wallet_address}")
        print("=" * 70)

        # Fetch balances (includes prices from Alchemy)
        print("\n1. Fetching token balances and prices...")
        balances = self.fetch_token_balances(wallet_address, network)

        if not balances:
            return {'error': 'No tokens found in wallet'}

        # Check if we need to fetch prices from CoinGecko as fallback
        missing_prices = [token['symbol'] for token in balances if token.get('price_usd') is None]

        if missing_prices:
            print(f"\n2. Fetching fallback prices for {len(missing_prices)} tokens...")
            fallback_prices = self.fetch_token_prices(missing_prices)
        else:
            print("\n2. All prices fetched from Alchemy!")
            fallback_prices = {}

        # Calculate composition
        print("\n3. Calculating portfolio composition...")
        portfolio_df = self.calculate_portfolio_composition(balances, fallback_prices)

        total_value = portfolio_df['value_usd'].sum()

        print(f"\n[OK] Total Portfolio Value: ${total_value:,.2f}")
        print("\nPortfolio Composition:")
        print(portfolio_df[['symbol', 'balance', 'price_usd', 'value_usd', 'weight']].to_string(index=False))

        # Identify risk assets vs stable assets
        stable_symbols = ['USDC', 'USDT', 'DAI', 'BUSD', 'FRAX']

        # Explicitly exclude major crypto assets that should never be considered stable
        non_stable_assets = {'ETH', 'BTC', 'WETH', 'WBTC', 'LINK', 'UNI', 'AAVE', 'MKR',
                            'SNX', 'COMP', 'YFI', 'SUSHI', 'CRV', 'BAL', 'MATIC', 'AVAX',
                            'SOL', 'DOT', 'ADA', 'BNB', 'XRP', 'DOGE', 'SHIB', 'ARB', 'OP'}

        try:
            from ..utils.cache import get_cache
            cache = get_cache()

            # Get stablecoin data from cache or API (24h TTL)
            stables = cache.get_or_set(
                'stablecoins',
                lambda: coingecko_api(coins='stablecoins', url="FDV", use_cached_data=False),
                ttl_seconds=86400  # 24 hours
            )

            # Convert all symbols to uppercase for case-insensitive matching
            # Filter out non-stable assets
            stable_symbols_from_api = [
                s.upper() for s in stables['symbol'].tolist()
                if s and s.upper() not in non_stable_assets
            ]

            stable_symbols.extend(stable_symbols_from_api)
            print(f'[OK] Loaded {len(stable_symbols_from_api)} stable symbols from cache/API')

        except Exception as e:
            print(f'[!] Could not fetch dynamic stable symbols from CoinGecko: {e}')

        # Remove duplicates and ensure all uppercase
        stable_symbols = list(set([s.upper() for s in stable_symbols if s.upper() not in non_stable_assets]))

        print(f'\nKnown stablecoin symbols ({len(stable_symbols)} total): {stable_symbols[:20]}...')
        print(f'portfolio_df symbols: {portfolio_df["symbol"].tolist()}')

        # Convert portfolio symbols to uppercase for comparison
        portfolio_df['symbol_upper'] = portfolio_df['symbol'].str.upper()

        risk_assets = portfolio_df[~portfolio_df['symbol_upper'].isin(stable_symbols)]['value_usd'].sum()
        stable_assets = portfolio_df[portfolio_df['symbol_upper'].isin(stable_symbols)]['value_usd'].sum()

        print(f'\nrisk_assets: {portfolio_df[~portfolio_df['symbol_upper'].isin(stable_symbols)]['symbol_upper']}, stable_assets: {portfolio_df[portfolio_df['symbol_upper'].isin(stable_symbols)]['symbol_upper']}')

        risk_allocation = risk_assets / total_value if total_value > 0 else 0
        stable_allocation = stable_assets / total_value if total_value > 0 else 0

        print(f"\nAsset Allocation:")
        print(f"  Risk Assets (Crypto): {risk_allocation*100:.1f}% (${risk_assets:,.2f})")
        print(f"  Stable Assets (Stablecoins): {stable_allocation*100:.1f}% (${stable_assets:,.2f})")

        return {
            'wallet_address': wallet_address,
            'total_value_usd': total_value,
            'portfolio': portfolio_df.to_dict('records'),
            'risk_allocation': risk_allocation,
            'stable_allocation': stable_allocation,
            'top_holdings': portfolio_df.head(5)[['symbol', 'value_usd', 'weight']].to_dict('records')
        }

    def get_regime_recommendation(
        self,
        wallet_analysis: Dict,
        classifier: "MacroRegimeClassifier"
    ) -> Dict:
        """
        Get investment recommendation based on current regime and portfolio.

        Args:
            wallet_analysis: Result from analyze_wallet()
            classifier: Trained MacroRegimeClassifier

        Returns:
            Dictionary with recommendation
        """
        # Get current regime
        regime_result = classifier.predict_current_regime(verbose=False)

        current_regime = regime_result['regime']
        confidence = regime_result['confidence']
        risk_on_prob = regime_result['risk_on_probability']

        risk_allocation = wallet_analysis['risk_allocation']

        # Recommendation logic
        if current_regime == 'Risk-On':
            optimal_risk = 0.80  # 80% in risk assets during Risk-On
            if risk_allocation < 0.5:
                action = 'INCREASE'
                suggestion = f"Consider increasing crypto exposure. Current: {risk_allocation*100:.0f}%, Optimal: {optimal_risk*100:.0f}%"
            elif risk_allocation > 0.95:
                action = 'MAINTAIN'
                suggestion = f"Portfolio well-positioned for Risk-On regime. Maintain current allocation."
            else:
                action = 'MAINTAIN'
                suggestion = f"Good allocation for Risk-On regime. Current exposure: {risk_allocation*100:.0f}%"

        else:  # Risk-Off
            optimal_risk = 0.20  # 20% in risk assets during Risk-Off
            if risk_allocation > 0.5:
                action = 'DECREASE'
                suggestion = f"Consider reducing crypto exposure. Current: {risk_allocation*100:.0f}%, Optimal: {optimal_risk*100:.0f}%"
            elif risk_allocation < 0.1:
                action = 'MAINTAIN'
                suggestion = f"Portfolio well-protected for Risk-Off regime. Maintain defensive position."
            else:
                action = 'MAINTAIN'
                suggestion = f"Reasonable allocation for Risk-Off regime. Current exposure: {risk_allocation*100:.0f}%"

        print("\n" + "=" * 70)
        print("REGIME-BASED RECOMMENDATION")
        print("=" * 70)
        print(f"Current Regime: {current_regime}")
        print(f"Confidence: {confidence*100:.1f}%")
        print(f"Risk-On Probability: {risk_on_prob*100:.1f}%")
        print(f"\nCurrent Risk Allocation: {risk_allocation*100:.1f}%")
        print(f"Recommended Action: {action}")
        print(f"\n{suggestion}")

        return {
            'regime': current_regime,
            'confidence': confidence,
            'risk_on_probability': risk_on_prob,
            'current_risk_allocation': risk_allocation,
            'optimal_risk_allocation': optimal_risk,
            'action': action,
            'suggestion': suggestion
        }


def main():
    """Test wallet analyzer with a sample address."""
    # Import here to avoid circular dependency
    from ..models import MacroRegimeClassifier

    # Example: Vitalik's address
    vitalik_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

    analyzer = WalletAnalyzer()

    # Note: This requires ALCHEMY_API_KEY to be set
    try:
        analysis = analyzer.analyze_wallet(vitalik_address)

        # Load trained classifier
        classifier = MacroRegimeClassifier()
        classifier.load('models/regime_classifier.pkl')

        # Get recommendation
        recommendation = analyzer.get_regime_recommendation(analysis, classifier)

    except Exception as e:
        print(f"[ERROR] {e}")
        print("\nTo use wallet analyzer, set ALCHEMY_API_KEY in .env file:")
        print("ALCHEMY_API_KEY=your_alchemy_api_key_here")
        print("\nGet your free API key at: https://www.alchemy.com/")


if __name__ == '__main__':
    main()
