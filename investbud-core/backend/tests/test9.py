"""
test9.py - Wallet portfolio tracker with accurate historical ETH balances

Gets all transfers from Alchemy API (erc20, external, internal).
For ETH, fetches actual on-chain balance at each transfer block using
eth_getBalance - this gives accurate historical balances that already
account for gas spent up to that point.

For other tokens, uses cumulative transfer balances (no gas involved).
"""

import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")
NETWORK = "eth-mainnet"
WALLET_ADDRESS = "0x27238ff1bf4ef450af1214a95963b9a92b351f08"


def hex_to_int(hex_str):
    """Convert hex string to int. Returns None if invalid."""
    if hex_str is None:
        return None
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return None


def get_eth_balance(address: str, block: str = "latest") -> float:
    """
    Get ETH balance from the chain at a specific block.

    Args:
        address: Wallet address
        block: Block number (hex string like "0x123") or "latest"

    Returns:
        Balance in ETH
    """
    response = requests.post(
        f"https://{NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
        json={
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [address, block],
            "id": 1
        }
    )
    result = response.json()
    if "result" in result:
        balance_wei = int(result["result"], 16)
        return balance_wei / 1e18
    return 0.0


def get_eth_balances_at_blocks(address: str, blocks: list) -> dict:
    """
    Get ETH balances at multiple blocks using batch requests.

    Args:
        address: Wallet address
        blocks: List of block numbers (integers)

    Returns:
        Dict mapping block_number -> balance_eth
    """
    balances = {}
    batch_size = 100

    print(f"    Fetching ETH balances at {len(blocks)} blocks...")

    for i in range(0, len(blocks), batch_size):
        batch = blocks[i:i + batch_size]

        batch_request = [
            {
                "jsonrpc": "2.0",
                "method": "eth_getBalance",
                "params": [address, hex(block)],
                "id": idx
            }
            for idx, block in enumerate(batch)
        ]

        response = requests.post(
            f"https://{NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
            json=batch_request
        )

        if response.status_code == 200:
            results = response.json()
            for idx, result in enumerate(results):
                if "result" in result:
                    block_num = batch[idx]
                    balance_wei = int(result["result"], 16)
                    balances[block_num] = balance_wei / 1e18

        if i > 0 and i % 500 == 0:
            print(f"      Processed {i}/{len(blocks)} blocks...")

    print(f"      Got {len(balances)} balances")
    return balances


def get_asset_transfers(address: str, direction: str = "to") -> list:
    """
    Fetch ALL asset transfers for an address with full pagination.

    Args:
        address: Wallet address
        direction: "to" for inflows, "from" for outflows

    Returns:
        List of all transfer objects
    """
    all_transfers = []
    page_key = None
    page_count = 0

    print(f"    Fetching {direction} transfers...")

    while True:
        params = {
            "fromBlock": "0x0",
            "toBlock": "latest",
            "category": ["erc20", "external", "internal"],
            "withMetadata": True,
            "maxCount": "0x3e8",  # 1000 per page (max)
            "excludeZeroValue": False,  # Include zero-value transfers
        }

        if direction == "to":
            params["toAddress"] = address
        else:
            params["fromAddress"] = address

        if page_key:
            params["pageKey"] = page_key

        response = requests.post(
            f"https://{NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
            json={
                "jsonrpc": "2.0",
                "method": "alchemy_getAssetTransfers",
                "params": [params],
                "id": 1
            }
        )

        result = response.json()
        if "result" not in result:
            print(f"      [!] API error: {result}")
            break

        transfers = result["result"].get("transfers", [])
        all_transfers.extend(transfers)
        page_count += 1

        # Check for more pages
        page_key = result["result"].get("pageKey")
        if not page_key:
            break

        print(f"      Page {page_count}: {len(all_transfers)} transfers so far...")

    print(f"      Total: {len(all_transfers)} {direction} transfers ({page_count} pages)")
    return all_transfers


def parse_transfers(transfers: list, direction: str, seen_ids: set = None) -> list:
    """
    Parse raw transfer objects into structured rows.

    Args:
        transfers: List of transfer objects from Alchemy API
        direction: "to" for inflows, "from" for outflows
        seen_ids: Set of unique IDs for deduplication

    Returns:
        List of parsed transfer dictionaries
    """
    rows = []
    if seen_ids is None:
        seen_ids = set()

    for transfer in transfers:
        # Deduplicate by uniqueId (Alchemy provides this)
        unique_id = transfer.get("uniqueId")
        if unique_id in seen_ids:
            continue
        seen_ids.add(unique_id)

        block_number = hex_to_int(transfer.get("blockNum"))
        value = transfer.get("value") or 0

        # Get timestamp from metadata
        metadata = transfer.get("metadata") or {}
        block_timestamp = metadata.get("blockTimestamp")
        if block_timestamp:
            timestamp = datetime.fromisoformat(block_timestamp.replace("Z", "+00:00"))
        else:
            timestamp = None

        # Determine sign: inflow = positive, outflow = negative
        signed_value = value if direction == "to" else -value

        raw_contract = transfer.get("rawContract") or {}
        token_address = raw_contract.get("address")
        token_decimals = hex_to_int(raw_contract.get("decimal"))

        # For ETH, use standard values
        asset = transfer.get("asset") or "UNKNOWN"
        if asset == "ETH" and token_address is None:
            token_address = "0x0000000000000000000000000000000000000000"
            token_decimals = 18

        rows.append({
            "block_number": block_number,
            "timestamp": timestamp,
            "date": timestamp.date() if timestamp else None,
            "tx_hash": transfer.get("hash"),
            "from_address": transfer.get("from"),
            "to_address": transfer.get("to"),
            "direction": direction,
            "value": signed_value,
            "symbol": asset,
            "token_address": token_address,
            "token_decimals": token_decimals,
            "category": transfer.get("category"),
        })

    return rows


def get_transaction_receipts_batch(tx_hashes: list) -> list:
    """
    Fetch transaction receipts in batches using JSON-RPC batch requests.

    Args:
        tx_hashes: List of transaction hashes

    Returns:
        List of receipt objects
    """
    receipts = []
    batch_size = 100

    print(f"    Fetching {len(tx_hashes)} transaction receipts...")

    for i in range(0, len(tx_hashes), batch_size):
        batch = tx_hashes[i:i + batch_size]

        batch_request = [
            {
                "jsonrpc": "2.0",
                "method": "eth_getTransactionReceipt",
                "params": [tx_hash],
                "id": idx
            }
            for idx, tx_hash in enumerate(batch)
        ]

        response = requests.post(
            f"https://{NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
            json=batch_request
        )

        if response.status_code == 200:
            results = response.json()
            for result in results:
                if "result" in result and result["result"]:
                    receipts.append(result["result"])

        if i > 0 and i % 500 == 0:
            print(f"      Processed {i}/{len(tx_hashes)} receipts...")

    print(f"      Got {len(receipts)} receipts")
    return receipts


def get_gas_costs_from_receipts(transfers_df: pd.DataFrame, wallet_address: str) -> tuple:
    """
    Get per-transaction gas costs by fetching receipts.

    Only counts gas for transactions where wallet is the TRANSACTION SENDER
    (the one who signed and paid gas), not just where wallet appears in transfers.

    Args:
        transfers_df: DataFrame with transfers
        wallet_address: The wallet address (lowercase)

    Returns:
        Tuple of (gas_df, total_gas_spent)
    """
    wallet_address = wallet_address.lower()

    # Get ALL unique tx hashes (we need to check who actually paid gas)
    all_tx_hashes = transfers_df["tx_hash"].unique()

    if len(all_tx_hashes) == 0:
        return pd.DataFrame(columns=["date", "timestamp", "tx_hash", "gas_eth"]), 0.0

    # Fetch receipts
    receipts = get_transaction_receipts_batch(list(all_tx_hashes))

    # Build gas costs dataframe
    gas_records = []
    total_gas = 0.0

    # Create tx_hash -> (date, timestamp) mapping from transfers
    tx_metadata = transfers_df[["tx_hash", "date", "timestamp"]].drop_duplicates("tx_hash")
    tx_metadata = tx_metadata.set_index("tx_hash")

    for receipt in receipts:
        tx_hash = receipt.get("transactionHash")

        # CRITICAL: Only count gas if wallet is the TRANSACTION sender (paid gas)
        # The receipt's "from" field is who submitted/signed the tx and paid gas
        tx_sender = receipt.get("from", "").lower()
        if tx_sender != wallet_address:
            continue  # Someone else paid gas for this tx

        gas_used = hex_to_int(receipt.get("gasUsed")) or 0
        effective_gas_price = hex_to_int(receipt.get("effectiveGasPrice")) or 0

        # Calculate gas in ETH
        gas_eth = (gas_used * effective_gas_price) / 1e18
        total_gas += gas_eth

        # Get date/timestamp from transfer metadata
        if tx_hash in tx_metadata.index:
            meta = tx_metadata.loc[tx_hash]
            gas_records.append({
                "date": meta["date"],
                "timestamp": meta["timestamp"],
                "tx_hash": tx_hash,
                "gas_eth": gas_eth
            })

    gas_df = pd.DataFrame(gas_records)
    return gas_df, total_gas


def calculate_daily_balances(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative daily balances per token."""
    if df.empty:
        return pd.DataFrame()

    # Group by date and token, sum the values
    daily = df.groupby(["date", "token_address", "symbol"]).agg({
        "value": "sum",
        "token_decimals": "first"
    }).reset_index()

    # Calculate cumulative balance per token
    daily = daily.sort_values(["token_address", "date"])
    daily["balance"] = daily.groupby("token_address")["value"].cumsum()

    return daily


def get_historical_prices(token_address: str, start_time: str, end_time: str) -> dict:
    """
    Fetch historical prices for a token (handles 365-day limit).

    Args:
        token_address: Token contract address
        start_time: ISO format datetime string
        end_time: ISO format datetime string

    Returns:
        Dict mapping date -> price
    """
    prices = {}

    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

    max_days = 364
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=max_days), end_dt)

        chunk_start = current_start.isoformat().replace("+00:00", "Z")
        chunk_end = current_end.isoformat().replace("+00:00", "Z")

        # Build payload
        if token_address == "0x0000000000000000000000000000000000000000":
            payload = {
                "symbol": "ETH",
                "startTime": chunk_start,
                "endTime": chunk_end,
                "interval": "1d"
            }
        else:
            payload = {
                "network": NETWORK,
                "address": token_address,
                "startTime": chunk_start,
                "endTime": chunk_end,
                "interval": "1d"
            }

        response = requests.post(
            f"https://api.g.alchemy.com/prices/v1/{ALCHEMY_API_KEY}/tokens/historical",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            data = result.get("data", result)
            if isinstance(data, list):
                price_list = data
            elif isinstance(data, dict):
                price_list = data.get("prices", [])
            else:
                price_list = []

            for price_point in price_list:
                ts = price_point.get("timestamp")
                val = price_point.get("value")
                if ts and val:
                    try:
                        date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
                        prices[date] = float(val)
                    except (ValueError, TypeError):
                        continue
        else:
            print(f"      [!] Price API error: {response.status_code}")

        current_start = current_end + timedelta(days=1)

    return prices


def main():
    print(f"[*] Wallet Portfolio Tracker - Simple Transfer-Based")
    print(f"    Address: {WALLET_ADDRESS}")
    print()

    # 1. Fetch all transfers
    print("[1] Fetching transfers from Alchemy API...")
    inflows = get_asset_transfers(WALLET_ADDRESS, direction="to")
    outflows = get_asset_transfers(WALLET_ADDRESS, direction="from")

    # 2. Parse and combine transfers (with deduplication)
    print("\n[2] Parsing transfers...")
    seen_ids = set()
    rows = []
    rows.extend(parse_transfers(inflows, "to", seen_ids))
    rows.extend(parse_transfers(outflows, "from", seen_ids))
    print(f"    Unique transfers: {len(seen_ids)}")

    df = pd.DataFrame(rows)

    if df.empty:
        print("[!] No transfers found")
        return

    # Filter out spam tokens
    spam_indicators = ["claim", "reward", "visit", "airdrop", ".org", ".com", ".io"]
    original_count = len(df)
    df = df[~df["symbol"].str.lower().str.contains("|".join(spam_indicators), na=False)]
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"    Filtered {filtered_count} spam transfers")

    print(f"    Final transfer count: {len(df)}")

    # 3. Show summary of transfers by category
    print("\n[3] Transfer Summary:")
    print(f"    Categories: {df['category'].value_counts().to_dict()}")
    print(f"    Directions: {df['direction'].value_counts().to_dict()}")
    print(f"    Tokens: {df['symbol'].nunique()} unique tokens")

    # Show transfers by token
    print("\n    By Token:")
    for symbol in df["symbol"].unique()[:10]:  # Top 10 tokens
        token_df = df[df["symbol"] == symbol]
        inflow = token_df[token_df["value"] > 0]["value"].sum()
        outflow = abs(token_df[token_df["value"] < 0]["value"].sum())
        net = inflow - outflow
        print(f"      {symbol}: +{inflow:.6f} -{outflow:.6f} = {net:.6f}")

    # 4. Calculate daily balances per token
    print("\n[4] Calculating daily balances...")
    daily_balances = calculate_daily_balances(df)
    print(f"    Days with activity: {daily_balances['date'].nunique()}")

    # 5. Get actual ETH balances at each ETH transfer block
    print("\n[5] Fetching actual ETH balances at transfer blocks...")
    eth_address = "0x0000000000000000000000000000000000000000"

    # Get unique blocks where ETH was transferred
    eth_transfers = df[df["token_address"] == eth_address].copy()

    if not eth_transfers.empty:
        eth_blocks = eth_transfers["block_number"].dropna().unique().astype(int).tolist()
        eth_blocks = sorted(set(eth_blocks))  # Dedupe and sort

        # Fetch actual balances at each block
        block_balances = get_eth_balances_at_blocks(WALLET_ADDRESS, eth_blocks)

        # Create mapping: block -> (date, actual_balance)
        block_to_date = eth_transfers.set_index("block_number")["date"].to_dict()

        # Build ETH daily balances using actual on-chain values
        eth_daily_actual = []
        for block in sorted(block_balances.keys()):
            date = block_to_date.get(block)
            if date:
                eth_daily_actual.append({
                    "date": date,
                    "token_address": eth_address,
                    "symbol": "ETH",
                    "balance": block_balances[block],
                    "block_number": block
                })

        if eth_daily_actual:
            eth_df = pd.DataFrame(eth_daily_actual)
            # If multiple ETH transfers on same date, take the last block's balance
            eth_df = eth_df.sort_values("block_number").drop_duplicates("date", keep="last")
            eth_df = eth_df.drop(columns=["block_number"])

            # Replace ETH entries in daily_balances with actual balances
            daily_balances = daily_balances[daily_balances["token_address"] != eth_address]
            daily_balances = pd.concat([daily_balances, eth_df], ignore_index=True)
            daily_balances = daily_balances.sort_values(["token_address", "date"])

            # Calculate gas spent (difference between transfer-calculated and actual)
            calculated_eth = calculate_daily_balances(df)
            calculated_final = calculated_eth[
                (calculated_eth["token_address"] == eth_address)
            ]["balance"].iloc[-1] if not calculated_eth[calculated_eth["token_address"] == eth_address].empty else 0

            actual_final = eth_df["balance"].iloc[-1]
            gas_spent = calculated_final - actual_final
            print(f"    Calculated ETH (from transfers): {calculated_final:.8f}")
            print(f"    Actual ETH (on-chain):           {actual_final:.8f}")
            print(f"    Total gas spent:                 {gas_spent:.8f} ETH")
    else:
        print("    No ETH transfers found")
        gas_spent = 0

    # Show current balances
    latest_date = daily_balances["date"].max()
    print("\n    Current Balances (actual on-chain):")
    for _, row in daily_balances[daily_balances["date"] == latest_date].iterrows():
        if abs(row["balance"]) > 0.000001:
            print(f"      {row['symbol']}: {row['balance']:.8f}")

    # 6. Get date range and fetch prices
    tokens = df[["token_address", "symbol"]].drop_duplicates()
    min_date = df["date"].min()
    max_date = datetime.now().date()

    start_time = datetime.combine(min_date, datetime.min.time()).isoformat() + "Z"
    end_time = datetime.combine(max_date, datetime.max.time()).isoformat() + "Z"

    print(f"\n[6] Fetching historical prices...")
    print(f"    Date range: {min_date} to {max_date}")
    print(f"    Tokens: {len(tokens)}")

    price_data = {}
    for _, token in tokens.iterrows():
        token_addr = token["token_address"]
        symbol = token["symbol"]
        if token_addr:
            print(f"    Fetching: {symbol}...")
            price_data[token_addr] = get_historical_prices(token_addr, start_time, end_time)

    # 7. Build daily portfolio value timeseries
    print("\n[7] Building portfolio timeseries...")

    date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    portfolio_rows = []

    for token_addr in tokens["token_address"].unique():
        if not token_addr:
            continue

        token_df = daily_balances[daily_balances["token_address"] == token_addr].copy()
        if token_df.empty:
            continue

        symbol = token_df["symbol"].iloc[0]
        prices = price_data.get(token_addr, {})

        # Create full date range for this token
        token_ts = pd.DataFrame({"date": date_range.date})
        token_ts = token_ts.merge(
            token_df[["date", "balance"]],
            on="date",
            how="left"
        )

        # Forward fill balances
        token_ts["balance"] = token_ts["balance"].ffill().fillna(0)

        # Add prices
        token_ts["price_usd"] = token_ts["date"].map(prices).fillna(0)
        token_ts["value_usd"] = token_ts["balance"] * token_ts["price_usd"]
        token_ts["token_address"] = token_addr
        token_ts["symbol"] = symbol

        portfolio_rows.append(token_ts)

    if not portfolio_rows:
        print("[!] No portfolio data to display")
        return

    portfolio_df = pd.concat(portfolio_rows, ignore_index=True)

    # Calculate total portfolio value per day
    total_value = portfolio_df.groupby("date").agg({
        "value_usd": "sum"
    }).reset_index()
    total_value.columns = ["date", "total_portfolio_usd"]

    # 7. Show final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS (Gas-Adjusted)")
    print("=" * 60)

    # Current balances with USD values
    latest = portfolio_df[portfolio_df["date"] == portfolio_df["date"].max()]
    latest_with_value = latest[latest["value_usd"] > 0.01].sort_values("value_usd", ascending=False)

    print("\n[Final Balances]")
    for _, row in latest_with_value.iterrows():
        print(f"  {row['symbol']:8} {row['balance']:>14.8f}  ${row['value_usd']:>10.2f}")

    # Total portfolio value
    total_usd = latest["value_usd"].sum()
    print(f"\n  {'TOTAL':8} {'':>14}  ${total_usd:>10.2f}")

    # Show gas summary
    print(f"\n[Gas Summary]")
    print(f"  Total gas spent: {gas_spent:.8f} ETH")

    # Show all balances including zero-value
    print("\n[All Token Balances (gas-adjusted)]")
    for _, row in latest.iterrows():
        if abs(row["balance"]) > 0.000001:
            print(f"  {row['symbol']:8} {row['balance']:>14.8f}")

    # 8. Save outputs
    print("\n[8] Saving outputs...")
    portfolio_df.to_csv("test9_portfolio_timeseries.csv", index=False)
    total_value.to_csv("test9_portfolio_total.csv", index=False)
    df.to_csv("test9_transfers.csv", index=False)

    print("    - test9_transfers.csv")
    print("    - test9_portfolio_timeseries.csv")
    print("    - test9_portfolio_total.csv")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
