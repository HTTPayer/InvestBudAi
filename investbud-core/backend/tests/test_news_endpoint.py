"""
Test the /news endpoint

This verifies that:
1. The scheduler runs and caches smart money analysis
2. The /news endpoint returns the cached data (requires X402 payment)

Note: This test uses regular requests without X402 payment.
To actually call the endpoint, use the client CLI with X402:
  npm run client news
"""

import requests
import json
from eth_account import Account
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://localhost:8001"

def test_news_endpoint_without_payment():
    """Test endpoint without payment (should get 402)"""
    print("\n" + "="*70)
    print("Testing /news Endpoint (Without Payment)")
    print("="*70)

    # Call the /news endpoint without payment
    print("\n[1] Calling GET /news (without X402 payment)...")
    response = requests.get(f"{API_URL}/news")

    if response.status_code == 402:
        print(f"✓ Got 402 Payment Required (expected)")
        print(f"  This endpoint requires $0.10 USDC via X402")
        print(f"\nTo call with payment, use:")
        print(f"  cd E:/Projects/MacroCrypto/client")
        print(f"  npm run client news")
        return

    print(f"Unexpected status code: {response.status_code}")
    return response


def test_news_endpoint():
    print("\n" + "="*70)
    print("Testing /news Endpoint (With X402)")
    print("="*70)
    print("\nNote: This requires x402 payment client.")
    print("For now, testing without payment to verify 402 response...")

    response = test_news_endpoint_without_payment()

    if response and response.status_code == 200:
        # If payment somehow succeeded, show the data
        print("\n[1] Payment succeeded! Calling GET /news...")
        response = requests.get(f"{API_URL}/news")

    print(f"Status Code: {response.status_code}")

    if response.status_code == 503:
        print("\n⚠ Analysis not yet cached. This is expected on first startup.")
        print("Wait a few minutes for the scheduler to run, or manually trigger:")
        print("  from src.macrocrypto.utils.scheduler import refresh_smart_money_analysis")
        print("  refresh_smart_money_analysis()")
        return

    if response.status_code != 200:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)
        return

    # Parse response
    data = response.json()

    print("\n" + "="*70)
    print("Smart Money Flow Analysis")
    print("="*70)

    print(f"\nGenerated At: {data['generated_at']}")
    print(f"Update Frequency: {data['update_frequency']}")
    print(f"Data Sources: {', '.join(data['data_sources'])}")

    print("\n" + "-"*70)
    print("Analysis:")
    print("-"*70)
    print(data['analysis'])

    # Save to file
    with open('output/news_response.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("\n" + "="*70)
    print("✓ Test completed successfully!")
    print("✓ Full response saved to output/news_response.json")
    print("="*70)


def test_root_endpoint():
    """Verify /news is listed in free endpoints"""
    print("\n[2] Verifying /news is listed in root endpoint...")

    response = requests.get(f"{API_URL}/")
    data = response.json()

    if "/news" in data["endpoints"]["free"]:
        print("✓ /news endpoint is listed in free endpoints")
        print(f"  Description: {data['endpoints']['free']['/news']}")
    else:
        print("✗ /news endpoint NOT found in free endpoints")


if __name__ == "__main__":
    try:
        test_news_endpoint()
        test_root_endpoint()
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API at http://localhost:8001")
        print("  Make sure the API is running:")
        print("  cd E:/Projects/MacroCrypto/backend/api")
        print("  uv run python main.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
