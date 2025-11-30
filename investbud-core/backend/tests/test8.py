"""
Multi-API Orchestration Demo (Python version of demo_06.ts)
Calls: Nansen → Heurist → LLM Chat → Translation
Powered by X402 payment rails
"""

from eth_account import Account
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from x402.clients.requests import x402_requests

load_dotenv()

# Configuration
PRIVATE_KEY = os.getenv("CLIENT_PRIVATE_KEY")
LLM_SERVER = os.getenv("LLM_SERVER", "https://api.httpayer.com/llm")
SERVER_API_KEY = os.getenv("SERVER_API_KEY", "")
RELAY_URL = "https://relay.httpayer.com"

if not PRIVATE_KEY:
    raise ValueError("PRIVATE_KEY not found in .env")

# Initialize account
account = Account.from_key(PRIVATE_KEY)
print(f"\n{'='*70}")
print(f"Multi-API Orchestration Demo")
print(f"{'='*70}")
print(f"Account: {account.address}")
print(f"Chain: Base")
print()

# Create X402 session
session = x402_requests(account)


def log_step(step: int, message: str):
    """Log a step in the process"""
    print(f"\n[Step {step}] {message}")
    print("-" * 70)


def save_to_file(data: dict, filename: str):
    """Save data to output directory"""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = output_dir / f"{filename}_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved to: {filepath}")
    return filepath


def get_nansen_data(session):
    """
    Step 1: Get Nansen Smart Money netflow data
    """
    log_step(1, "Getting Nansen Smart Money data")

    TARGET_API = "https://nansen.api.corbits.dev/api/v1/smart-money/netflow"

    request_data = {
        "chains": ["all"],
        "filters": {
            "include_native_tokens": True,
            "include_smart_money_labels": ["Fund", "Smart Trader"],
            "include_stablecoins": True
        },
        "pagination": {
            "page": 1,
            "per_page": 100
        },
        "order_by": [
            {
                "field": "net_flow_24h_usd",
                "direction": "DESC"
            }
        ]
    }

    payload = {
        "api_url": TARGET_API,
        "method": "POST",
        "network": "base",
        "data": request_data
    }

    print(f"Calling: {TARGET_API}")

    response = session.post(
        RELAY_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    # Check for payment response header
    payment_header = response.headers.get("x-payment-response")
    if payment_header:
        print(f"✓ Payment executed")

    data = response.json()
    print(f"✓ Nansen response received")

    save_to_file(data, "nansen_netflow")
    return data



def get_heurist_search(session):
    """
    Step 2: Get Heurist AI search results based on Nansen tokens
    """
    log_step(2, "Searching crypto news with Heurist AI")

    TARGET_API = "https://mesh.heurist.xyz/x402/agents/ExaSearchDigestAgent/exa_web_search"

    # Extract token symbols and sectors from Nansen data

    # Create search query
    search_term = f"Recent cryptocurrency news and market analysis for the DePIN and AI sectors."

    print(f"Calling: {TARGET_API}")
    print(f"Search term: {search_term[:100]}...")

    payload = {
        "api_url": TARGET_API,
        "method": "POST",
        "network": "base",
        "data": {
            "search_term": search_term,
            "limit": 5,
            "time_filter": "past_week"
        }
    }

    response = session.post(
        RELAY_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    # Check for payment response header
    payment_header = response.headers.get("x-payment-response")
    if payment_header:
        print(f"✓ Payment executed")

    data = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
    print(f"✓ Heurist response received")

    save_to_file({"search_term": search_term, "results": data}, "heurist_search")
    return data

def summarize_with_llm(session, nansen_data, heurist_data):
    """
    Step 3: Summarize with LLM (GPT-4)
    """
    log_step(3, "Generating analysis with LLM")

    TARGET_API = f"{LLM_SERVER}/chat"

    print(f"Calling: {TARGET_API}")

    request_data = {
        "messages": [
            {
                "role": "system",
                "content": "You are a research analyst that analyzes cryptocurrency smart money flows and related news to generate a concise and insightful report for defi traders."
            },
            {
                "role": "user",
                "content": f"""Analyze the following data:

SMART MONEY TOKEN FLOWS (from Nansen):
{json.dumps(nansen_data, indent=2)}

RELATED NEWS & MARKET ANALYSIS (from Heurist):
{json.dumps(heurist_data, indent=2)}

Provide a concise summary that:
1. Identifies which tokens smart money is accumulating or selling
2. Explains potential reasons based on the news articles
3. Highlights key trends or opportunities

Note: 
Whenever you mention a token for the first time, include its token_address and chain in parentheses.

"""
            }
        ]
    }

    headers = {
        "Content-Type": "application/json"
    }
    if SERVER_API_KEY:
        headers["x-api-key"] = SERVER_API_KEY

    response = session.post(
        TARGET_API,
        json=request_data,
        headers=headers
    )

    # Check for payment response header
    payment_header = response.headers.get("x-payment-response")
    if payment_header:
        print(f"✓ Payment executed")

    data = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
    print(f"✓ LLM response received")

    save_to_file(data, "llm_chat")
    return data


def extract_text(obj):
    """Extract text from response object"""
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return extract_text(parsed)
        except:
            return obj

    if isinstance(obj, dict):
        if 'response' in obj:
            return extract_text(obj['response'])
        if 'text' in obj:
            return extract_text(obj['text'])
        if 'content' in obj:
            return extract_text(obj['content'])
        if 'message' in obj:
            return extract_text(obj['message'])

    return json.dumps(obj, indent=2)


def run_demo():
    """Main execution function"""
    try:

        # Step 2: Get Heurist search results
        heurist_data = get_heurist_search(session)

        print(heurist_data)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
