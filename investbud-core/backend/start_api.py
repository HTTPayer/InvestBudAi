"""
Start the MacroCrypto API server.
Run with: uv run python start_api.py
"""

import uvicorn, os
from dotenv import load_dotenv
load_dotenv()

PORT = os.getenv("PORT", 8015)

if __name__ == "__main__":
    print("=" * 70)
    print("Starting MacroCrypto API Server")
    print("=" * 70)
    print("\nFREE Endpoints:")
    print("  GET  /health        - Health check")
    print("  GET  /latest_report - Cached regime + model performance")
    print("  GET  /historical    - Backtest results (CSV/JSON)")
    print("  GET  /model/metrics - Model accuracy and performance metrics")
    print("\nPAID Endpoints:")
    print("  GET  /regime      - $0.01 - Current macro regime")
    print("  POST /portfolio   - $0.05 - Wallet analysis (no LLM)")
    print("  POST /advise      - $0.10 - Full advisory (wallet + LLM)")
    print("  POST /chat        - $0.02 - Stateful conversation (portfolio/regime)")
    print("\n" + "=" * 70)
    print(f"Starting server on http://0.0.0.0:{PORT}")
    print("=" * 70 + "\n")

    uvicorn.run("api.main:app", host="0.0.0.0", port=int(PORT), reload=True)
