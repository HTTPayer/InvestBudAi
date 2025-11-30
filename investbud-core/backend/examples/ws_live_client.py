#!/usr/bin/env python3
"""
x402 WebSocket Client for MacroCrypto Live Portfolio Streaming

Uses the actual x402 client library for payment at WebSocket handshake.

Usage:
    python examples/ws_live_client.py --address 0x742d35Cc6634C0532925a3b844Bc9e7595f2bD45

Requirements:
    pip install websockets aiohttp
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Optional

try:
    import websockets
    import aiohttp
except ImportError as e:
    print(f"Missing: pip install websockets aiohttp")
    sys.exit(1)

from dotenv import load_dotenv
from eth_account import Account
from x402.clients.requests import x402_requests

load_dotenv()


class X402WebSocketClient:
    """WebSocket client with x402 payment at handshake."""

    def __init__(
        self,
        base_url: str,
        wallet_address: str,
        private_key: str,
        network: str = "eth-mainnet",
    ):
        self.base_url = base_url.rstrip("/")
        self.wallet_address = wallet_address
        self.network = network
        self.ws = None
        self.session_info = None

        # Initialize x402 client
        self.account = Account.from_key(private_key)
        self.x402_session = x402_requests(self.account)
        print(f"[x402] Account: {self.account.address}")

    def _ws_url(self) -> str:
        """Build WebSocket URL."""
        base = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        return f"{base}/wallet/live?address={self.wallet_address}&network={self.network}"

    def _http_url(self) -> str:
        """HTTP URL for payment."""
        return f"{self.base_url}/wallet/live?address={self.wallet_address}&network={self.network}"

    async def connect(self) -> None:
        """
        Connect using x402 payment.

        Flow:
        1. Use x402_requests to make HTTP request (handles 402 payment)
        2. The payment receipt from successful request is reused for WebSocket
        """
        print("1. Making x402 payment via HTTP...")
        url = self._http_url()

        # x402_requests handles: request -> 402 -> pay -> retry with receipt
        response = self.x402_session.get(url)

        if response.status_code != 200:
            raise Exception(f"Payment failed: {response.status_code}")

        print("   ✓ Payment successful")

        # Get the receipt that was used for the successful request
        receipt = response.request.headers.get("X-PAYMENT", "")

        if not receipt:
            print("   ! No receipt found, trying WebSocket anyway...")

        print("2. Connecting to WebSocket...")
        ws_url = self._ws_url()

        # For WebSocket, we need to handle 402 differently since websockets
        # library doesn't support x402 directly. We'll pass the receipt as header.
        try:
            headers = {"X-PAYMENT": receipt} if receipt else {}
            self.ws = await websockets.connect(ws_url, additional_headers=headers)
        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 402:
                print("   ! WebSocket got 402, making separate payment...")
                # Make another payment specifically for WS
                response2 = self.x402_session.get(url)
                receipt2 = response2.request.headers.get("X-PAYMENT", "")
                headers = {"X-PAYMENT": receipt2} if receipt2 else {}
                self.ws = await websockets.connect(ws_url, additional_headers=headers)
            else:
                raise

        # Wait for connected message
        msg = await self.ws.recv()
        data = json.loads(msg)

        if data.get("type") == "error":
            raise Exception(f"Connection error: {data.get('message')}")

        if data.get("type") != "connected":
            raise Exception(f"Unexpected: {data}")

        self.session_info = data
        print("3. Connected!")
        print(f"   Session: {data.get('session_id')}")
        print(f"   Expires: {data.get('expires_at')}")
        print()

    async def stream(self) -> None:
        """Stream portfolio updates."""
        if not self.ws:
            raise Exception("Not connected")

        print("Streaming portfolio updates...")
        print("-" * 60)

        update_count = 0

        try:
            async for msg in self.ws:
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "portfolio_update":
                    update_count += 1
                    self._print_update(data, update_count)

                elif msg_type == "session_ending":
                    print(f"\n⚠️  Session ending in {data.get('seconds_remaining')}s")

                elif msg_type == "error":
                    if data.get("code") == "session_expired":
                        print("\n⏰ Session expired")
                        break
                    print(f"\n❌ Error: {data.get('message')}")

        except websockets.ConnectionClosed as e:
            print(f"\n🔌 Connection closed: {e.code} {e.reason}")

        print(f"\nSession complete: {update_count} updates")

    def _print_update(self, data: dict, count: int) -> None:
        """Pretty print portfolio update."""
        ts = data.get("timestamp", "")[:19]
        value = data.get("total_value_usd", 0)
        change = data.get("change_24h")
        positions = data.get("positions", [])

        change_str = ""
        if change is not None:
            arrow = "📈" if change >= 0 else "📉"
            change_str = f" {arrow} {change * 100:+.2f}%"

        print(f"[{ts}] #{count} 💰 ${value:,.2f}{change_str}")

        # Show all positions on every update
        for p in positions:
            print(f"   {p['symbol']}: ${p['value_usd']:,.2f} ({p['weight']*100:.1f}%)")

    async def disconnect(self) -> None:
        if self.ws:
            await self.ws.close()
            self.ws = None


async def main():
    parser = argparse.ArgumentParser(description="MacroCrypto Live Portfolio Client")
    parser.add_argument("--address", "-a", required=True, help="Wallet address (0x...)")
    parser.add_argument("--network", "-n", default="eth-mainnet", help="Network")
    parser.add_argument("--url", "-u", default=os.getenv("ORACLE_API_URL", "http://localhost:8015"))
    parser.add_argument("--private-key", "-k", default=os.getenv("CLIENT_PRIVATE_KEY"))
    args = parser.parse_args()

    if not args.private_key:
        print("❌ Missing private key")
        print("   Set CLIENT_PRIVATE_KEY env var or use --private-key")
        sys.exit(1)

    if not args.address.startswith("0x") or len(args.address) != 42:
        print("❌ Invalid wallet address")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  MacroCrypto Live Portfolio Stream")
    print("=" * 60)
    print(f"  Wallet:  {args.address}")
    print(f"  Network: {args.network}")
    print(f"  API:     {args.url}")
    print("=" * 60)
    print()

    client = X402WebSocketClient(
        base_url=args.url,
        wallet_address=args.address,
        private_key=args.private_key,
        network=args.network,
    )

    try:
        await client.connect()
        await client.stream()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted")
    except Exception as e:
        print(f"\n❌ {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
