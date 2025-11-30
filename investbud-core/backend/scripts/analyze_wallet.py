"""
Analyze a crypto wallet: fetch holdings, calculate metrics, get recommendations.
Run with: uv run python analyze_wallet.py <wallet_address>
"""
import sys
from src.macrocrypto.data import WalletAnalyzer
from src.macrocrypto.models import MacroRegimeClassifier


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python analyze_wallet.py <wallet_address>")
        print("\nExample addresses to try:")
        print("  Vitalik: 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045")
        print("  Example: 0x47ac0Fb4F2D84898e4D9E7b4DaB3C24507a6D503")
        return

    wallet_address = sys.argv[1]

    # Check if address looks valid
    if not wallet_address.startswith('0x') or len(wallet_address) != 42:
        print(f"[ERROR] Invalid Ethereum address: {wallet_address}")
        print("Address should start with '0x' and be 42 characters long")
        return

    print("=" * 70)
    print("MacroCrypto Wallet Analyzer")
    print("=" * 70)

    # Initialize analyzer
    analyzer = WalletAnalyzer()

    # Analyze wallet
    try:
        print(f"\nAnalyzing wallet: {wallet_address}")
        analysis = analyzer.analyze_wallet(wallet_address)

        if 'error' in analysis:
            print(f"\n[ERROR] {analysis['error']}")
            return

        # Load trained classifier
        print("\n4. Loading regime classifier...")
        classifier = MacroRegimeClassifier()
        try:
            classifier.load('models/regime_classifier.pkl')
        except FileNotFoundError:
            print("[ERROR] Trained model not found.")
            print("Please run 'uv run python train_classifier.py' first.")
            return

        # Get recommendation
        print("\n5. Generating recommendations...")
        recommendation = analyzer.get_regime_recommendation(analysis, classifier)

        # Summary
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Wallet: {wallet_address}")
        print(f"Total Value: ${analysis['total_value_usd']:,.2f}")
        print(f"Risk Allocation: {analysis['risk_allocation']*100:.1f}%")
        print(f"Stable Allocation: {analysis['stable_allocation']*100:.1f}%")
        print(f"\nCurrent Regime: {recommendation['regime']}")
        print(f"Confidence: {recommendation['confidence']*100:.1f}%")
        print(f"Recommended Action: {recommendation['action']}")
        print(f"\nTop Holdings:")
        for holding in analysis['top_holdings']:
            print(f"  {holding['symbol']}: ${holding['value_usd']:,.2f} ({holding['weight']*100:.1f}%)")

    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\nTo use wallet analyzer, you need an Alchemy API key.")
        print("1. Get a free API key at: https://www.alchemy.com/")
        print("2. Add it to your .env file:")
        print("   ALCHEMY_API_KEY=your_alchemy_api_key_here")

    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
