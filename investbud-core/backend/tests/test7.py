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
    """Convert hex string (e.g., '0x12') to int. Returns None if invalid."""
    if hex_str is None:
        return None
    try:
        return int(hex_str, 16)
    except (ValueError, TypeError):
        return None


def get_eth_balance_at_block(address: str, block_number: int) -> float:
    """
    Get ETH balance at a specific block using eth_getBalance.

    Args:
        address: Wallet address
        block_number: Block number (int)

    Returns:
        Balance in ETH (float)
    """
    block_hex = hex(block_number)
    response = requests.post(
        f"https://{NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
        json={
            "jsonrpc": "2.0",
            "method": "eth_getBalance",
            "params": [address, block_hex],
            "id": 1
        }
    )
    result = response.json().get("result", "0x0")
    balance_wei = hex_to_int(result) or 0
    return balance_wei / 1e18


def get_transaction_receipt(tx_hash: str) -> dict:
    """Fetch transaction receipt for a given tx hash."""
    response = requests.post(
        f"https://{NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
        json={
            "jsonrpc": "2.0",
            "method": "eth_getTransactionReceipt",
            "params": [tx_hash],
            "id": 1
        }
    )
    return response.json().get("result", {})


def is_transaction_successful(tx_hash: str) -> bool:
    """Check if a transaction was successful (not reverted/failed)."""
    receipt = get_transaction_receipt(tx_hash)
    if not receipt:
        return True  # If we can't get status, assume success
    status = receipt.get("status", "0x1")
    return status == "0x1"  # 0x1 = success, 0x0 = failure


def get_asset_transfers(address: str, direction: str = "to") -> list:
    """
    Fetch ALL asset transfers for an address (handles pagination).

    Args:
        address: Wallet address
        direction: "to" for inflows, "from" for outflows
    """
    all_transfers = []
    page_key = None

    while True:
        params = {
            "fromBlock": "0x0",  # From genesis to get full history
            "toBlock": "latest",
            "category": ["erc20", "external", "internal"],  # internal = ETH from contracts
            "withMetadata": True,
            "maxCount": "0x3e8",  # 1000 per page
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
            print(f"  [!] API error: {result}")
            break

        transfers = result["result"].get("transfers", [])
        all_transfers.extend(transfers)

        # Check for more pages
        page_key = result["result"].get("pageKey")
        if not page_key:
            break

        print(f"    Fetching next page ({len(all_transfers)} transfers so far)...")

    return all_transfers


def parse_transfers(transfers: list, direction: str, seen_ids: set = None, check_status: bool = True) -> list:
    """Parse transfers into rows with sign based on direction.

    Args:
        transfers: List of transfer objects from Alchemy API
        direction: "to" for inflows, "from" for outflows
        seen_ids: Set of unique IDs for deduplication
        check_status: If True, verify transaction success and skip failed ones
    """
    rows = []
    if seen_ids is None:
        seen_ids = set()

    # Cache for transaction status checks (to avoid duplicate API calls)
    status_cache = {}

    for i, transfer in enumerate(transfers):
        # Deduplicate by uniqueId (Alchemy provides this)
        unique_id = transfer.get("uniqueId")
        if unique_id in seen_ids:
            continue
        seen_ids.add(unique_id)

        tx_hash = transfer.get("hash")

        # Check transaction status for outflows (failed txs still show in API but didn't transfer)
        if check_status and direction == "from" and tx_hash:
            if tx_hash not in status_cache:
                status_cache[tx_hash] = is_transaction_successful(tx_hash)
                if not status_cache[tx_hash]:
                    print(f"    [!] Skipping failed tx: {tx_hash[:20]}...")
            if not status_cache[tx_hash]:
                continue  # Skip failed transactions

        block_number = hex_to_int(transfer.get("blockNum"))
        value = transfer.get("value") or 0

        # Get timestamp from metadata if available
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
            token_address = "0x0000000000000000000000000000000000000000"  # Native ETH
            token_decimals = 18

        rows.append({
            "block_number": block_number,
            "timestamp": timestamp,
            "date": timestamp.date() if timestamp else None,
            "tx_hash": tx_hash,
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


def get_historical_prices(token_address: str, start_time: str, end_time: str) -> dict:
    """
    Fetch historical prices for a token.
    Handles Alchemy's 365-day limit by chunking requests.

    Args:
        token_address: Token contract address
        start_time: ISO format datetime string
        end_time: ISO format datetime string

    Returns:
        Dict mapping date -> price
    """
    prices = {}

    # Parse dates
    start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

    # Chunk into 364-day periods (leave margin for API)
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
            print(f"  [!] Price API error for {token_address}: {response.status_code} - {response.text[:200]}")

        # Move to next chunk
        current_start = current_end + timedelta(days=1)

    return prices


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


def get_all_transactions_from_address(address: str) -> list:
    """
    Fetch ALL transactions initiated by an address using multiple methods.
    This includes transfers, approvals, contract interactions, and failed txs.
    """
    all_tx_hashes = set()

    # Method 1: Get asset transfers (to capture transfer tx hashes)
    print("    Fetching asset transfers...")
    page_key = None
    while True:
        params = {
            "fromBlock": "0x0",
            "toBlock": "latest",
            "fromAddress": address,
            "category": ["external", "internal", "erc20", "erc721", "erc1155", "specialnft"],
            "withMetadata": True,
            "maxCount": "0x3e8",
            "excludeZeroValue": False,  # Include zero-value transfers
        }
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
            break

        transfers = result["result"].get("transfers", [])
        for t in transfers:
            all_tx_hashes.add(t["hash"])

        page_key = result["result"].get("pageKey")
        if not page_key:
            break

    print(f"    Found {len(all_tx_hashes)} transactions from asset transfers")

    # Method 2: Use Alchemy's transaction receipts API to find more transactions
    # Get transaction count to know how many transactions were sent
    response = requests.post(
        f"https://{NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
        json={
            "jsonrpc": "2.0",
            "method": "eth_getTransactionCount",
            "params": [address, "latest"],
            "id": 1
        }
    )
    result = response.json()
    tx_count = hex_to_int(result.get("result", "0x0")) or 0
    print(f"    Total transactions initiated by address (nonce): {tx_count}")

    # If we're missing transactions, try to find them via block scanning
    # (This is a fallback - asset transfers should catch most)
    if len(all_tx_hashes) < tx_count:
        missing_count = tx_count - len(all_tx_hashes)
        print(f"    Note: {missing_count} transactions not captured via asset transfers")
        print(f"    (These are likely approve/contract calls without value transfer)")

    return list(all_tx_hashes)


def get_gas_spent_detailed(address: str, known_tx_hashes: set = None) -> tuple:
    """
    Calculate total gas spent by an address (ETH used for transaction fees).
    Returns (total_gas_eth, gas_per_day_dict) for accurate daily tracking.

    Args:
        address: Wallet address
        known_tx_hashes: Optional set of known tx hashes to include

    Returns:
        Tuple of (total_gas_eth, dict mapping date -> gas_spent)
    """
    total_gas_eth = 0.0
    gas_per_day = {}

    # Get all transactions from this address
    all_tx_hashes = set(get_all_transactions_from_address(address))

    # Also include any known tx hashes (from transfers we've already parsed)
    if known_tx_hashes:
        all_tx_hashes.update(known_tx_hashes)

    print(f"    Calculating gas for {len(all_tx_hashes)} unique transactions...")

    for i, tx_hash in enumerate(all_tx_hashes):
        if (i + 1) % 50 == 0:
            print(f"    Processing {i + 1}/{len(all_tx_hashes)}...")

        # Get transaction receipt for gas used
        receipt = get_transaction_receipt(tx_hash)
        if not receipt:
            continue

        # Verify this transaction was FROM our address (sender pays gas)
        tx_from = receipt.get("from", "").lower()
        if tx_from != address.lower():
            continue

        gas_used = hex_to_int(receipt.get("gasUsed", "0x0"))
        effective_gas_price = hex_to_int(receipt.get("effectiveGasPrice", "0x0"))

        if gas_used and effective_gas_price:
            gas_cost_wei = gas_used * effective_gas_price
            gas_cost_eth = gas_cost_wei / 1e18
            total_gas_eth += gas_cost_eth

            # Get block timestamp for daily tracking
            block_num = receipt.get("blockNumber")
            if block_num:
                # Fetch block to get timestamp
                block_resp = requests.post(
                    f"https://{NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
                    json={
                        "jsonrpc": "2.0",
                        "method": "eth_getBlockByNumber",
                        "params": [block_num, False],
                        "id": 1
                    }
                )
                block = block_resp.json().get("result", {})
                if block:
                    timestamp = hex_to_int(block.get("timestamp", "0x0"))
                    if timestamp:
                        date = datetime.fromtimestamp(timestamp).date()
                        gas_per_day[date] = gas_per_day.get(date, 0) + gas_cost_eth

    return total_gas_eth, gas_per_day


def get_gas_spent(address: str) -> float:
    """
    Calculate total gas spent by an address (ETH used for transaction fees).
    This is a wrapper for backward compatibility.
    """
    total_gas, _ = get_gas_spent_detailed(address)
    return total_gas


def main():
    print(f"[*] Fetching transfers for {WALLET_ADDRESS}")

    # 1. Get inflows and outflows
    print("[*] Fetching inflows (to address)...")
    inflows = get_asset_transfers(WALLET_ADDRESS, direction="to")
    print(f"    Found {len(inflows)} inflows")

    print("[*] Fetching outflows (from address)...")
    outflows = get_asset_transfers(WALLET_ADDRESS, direction="from")
    print(f"    Found {len(outflows)} outflows")

    # 2. Parse and combine transfers (with deduplication)
    seen_ids = set()
    rows = []
    rows.extend(parse_transfers(inflows, "to", seen_ids))
    rows.extend(parse_transfers(outflows, "from", seen_ids))
    print(f"    Deduplicated: {len(seen_ids)} unique transfers")

    df = pd.DataFrame(rows)

    if df.empty:
        print("[!] No transfers found")
        return

    # Filter out spam tokens (tokens with suspicious names)
    spam_indicators = ["claim", "reward", "visit", "airdrop", ".org", ".com", ".io"]
    df = df[~df["symbol"].str.lower().str.contains("|".join(spam_indicators), na=False)]

    print(f"\n[*] Total transfers after filtering: {len(df)}")
    print(df[["date", "direction", "symbol", "value"]].head(10))

    # Debug: Show all ETH transfers
    eth_df = df[df["symbol"] == "ETH"].copy()
    print(f"\n=== DEBUG: All ETH Transfers ({len(eth_df)}) ===")
    print(eth_df[["date", "direction", "category", "value", "tx_hash"]].to_string())

    # 3. Get initial ETH balance using eth_getBalance at the first transfer block
    print("\n[*] Getting initial ETH balance...")
    earliest_block = df["block_number"].min()
    if earliest_block:
        # Get balance at block BEFORE the first transfer
        initial_eth_balance = get_eth_balance_at_block(WALLET_ADDRESS, earliest_block - 1)
        print(f"    Initial ETH balance at block {earliest_block - 1}: {initial_eth_balance:.6f} ETH")
    else:
        initial_eth_balance = 0.0

    # 4. Calculate daily balances per token
    print("\n[*] Calculating daily balances...")
    daily_balances = calculate_daily_balances(df)
    print(daily_balances)

    # 4. Get unique tokens and date range
    tokens = df[["token_address", "symbol"]].drop_duplicates()
    min_date = df["date"].min()
    max_date = df["date"].max() or datetime.now().date()

    # Extend to today if needed
    if max_date < datetime.now().date():
        max_date = datetime.now().date()

    start_time = datetime.combine(min_date, datetime.min.time()).isoformat() + "Z"
    end_time = datetime.combine(max_date, datetime.max.time()).isoformat() + "Z"

    print(f"\n[*] Date range: {min_date} to {max_date}")
    print(f"[*] Fetching historical prices for {len(tokens)} tokens...")

    # 5. Fetch historical prices for each token
    price_data = {}
    for _, token in tokens.iterrows():
        token_addr = token["token_address"]
        symbol = token["symbol"]
        if token_addr:
            print(f"    Fetching prices for {symbol} ({token_addr[:10]}...)")
            price_data[token_addr] = get_historical_prices(token_addr, start_time, end_time)

    # 6. Build daily portfolio value timeseries
    print("\n[*] Building portfolio value timeseries...")

    # Create a date range
    date_range = pd.date_range(start=min_date, end=max_date, freq="D")

    # For each token, forward-fill balance and multiply by price
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

        # Forward fill balances (balance persists until next transfer)
        token_ts["balance"] = token_ts["balance"].ffill().fillna(0)

        # Add prices
        token_ts["price_usd"] = token_ts["date"].map(prices).fillna(0)
        token_ts["value_usd"] = token_ts["balance"] * token_ts["price_usd"]
        token_ts["token_address"] = token_addr
        token_ts["symbol"] = symbol

        portfolio_rows.append(token_ts)

    if portfolio_rows:
        portfolio_df = pd.concat(portfolio_rows, ignore_index=True)

        # Calculate total portfolio value per day
        total_value = portfolio_df.groupby("date").agg({
            "value_usd": "sum"
        }).reset_index()
        total_value.columns = ["date", "total_portfolio_usd"]

        # Calculate ETH balance adjustment using actual current balance
        print("\n[*] Adjusting ETH balance using actual on-chain balance...")

        # Get the actual current ETH balance from the blockchain
        actual_current_balance = get_eth_balance_at_block(WALLET_ADDRESS, 0x7fffffff)  # "latest" equivalent
        # Alternative: use latest
        response = requests.post(
            f"https://{NETWORK}.g.alchemy.com/v2/{ALCHEMY_API_KEY}",
            json={
                "jsonrpc": "2.0",
                "method": "eth_getBalance",
                "params": [WALLET_ADDRESS, "latest"],
                "id": 1
            }
        )
        result = response.json().get("result", "0x0")
        actual_current_balance = (hex_to_int(result) or 0) / 1e18

        print(f"    Actual current ETH balance: {actual_current_balance:.8f} ETH")

        # Get our calculated balance from transfers
        eth_mask = portfolio_df["symbol"] == "ETH"
        eth_rows = portfolio_df[eth_mask].copy()
        latest_eth_date = eth_rows["date"].max()
        calculated_balance = eth_rows[eth_rows["date"] == latest_eth_date]["balance"].iloc[0] if len(eth_rows) > 0 else 0

        print(f"    Calculated balance (from transfers): {calculated_balance:.8f} ETH")

        # The difference is gas spent + any other discrepancies
        total_adjustment = calculated_balance - actual_current_balance
        print(f"    Total adjustment needed (gas + other): {total_adjustment:.8f} ETH")

        # Add initial balance to our calculations
        if initial_eth_balance > 0:
            print(f"    Adding initial balance offset: {initial_eth_balance:.8f} ETH")

        # Approach: Anchor to actual current balance
        # Calculate what our final balance should be and distribute the adjustment proportionally
        # across all dates based on time (simpler: just subtract total adjustment from all dates)

        # Simple approach: Linear distribution of gas from start to end
        if len(eth_rows) > 0:
            min_eth_date = eth_rows["date"].min()
            max_eth_date = eth_rows["date"].max()
            total_days = (max_eth_date - min_eth_date).days + 1

            def calculate_adjusted_balance(row):
                days_elapsed = (row["date"] - min_eth_date).days
                # Linear interpolation of adjustment
                if total_days > 1:
                    adjustment_fraction = days_elapsed / (total_days - 1)
                else:
                    adjustment_fraction = 1.0
                adjustment = total_adjustment * adjustment_fraction
                # Add initial balance, subtract proportional gas
                return row["balance"] + initial_eth_balance - adjustment

            portfolio_df.loc[eth_mask, "balance"] = eth_rows.apply(
                calculate_adjusted_balance,
                axis=1
            )
            portfolio_df["value_usd"] = portfolio_df["balance"] * portfolio_df["price_usd"]

            # Verify final balance
            final_balance = portfolio_df.loc[eth_mask & (portfolio_df["date"] == max_eth_date), "balance"].iloc[0]
            print(f"    Adjusted final ETH balance: {final_balance:.8f} ETH")
            print(f"    Match with actual: {abs(final_balance - actual_current_balance) < 0.0001}")

        # Recalculate total portfolio value
        total_value = portfolio_df.groupby("date").agg({
            "value_usd": "sum"
        }).reset_index()
        total_value.columns = ["date", "total_portfolio_usd"]

        # Filter out tokens that never have any USD value
        tokens_with_value = portfolio_df.groupby("token_address")["value_usd"].sum()
        tokens_to_keep = tokens_with_value[tokens_with_value > 0].index
        portfolio_df = portfolio_df[portfolio_df["token_address"].isin(tokens_to_keep)]

        # Filter out leading dates with negative/zero total value
        daily_totals = portfolio_df.groupby("date")["value_usd"].sum()
        positive_dates = daily_totals[daily_totals > 0].index
        first_positive_date = positive_dates.min()
        portfolio_df = portfolio_df[portfolio_df["date"] >= first_positive_date]
        print(f"    Starting from first positive date: {first_positive_date}")

        # Recalculate total after filtering
        total_value = portfolio_df.groupby("date").agg({
            "value_usd": "sum"
        }).reset_index()
        total_value.columns = ["date", "total_portfolio_usd"]

        print("\n=== Final Balances (Gas-Adjusted) ===")
        latest = portfolio_df[portfolio_df["date"] == portfolio_df["date"].max()]
        print(latest[["symbol", "balance", "price_usd", "value_usd"]].sort_values("value_usd", ascending=False))

        # 7. Calculate composition (weights) per token over time
        print("\n[*] Calculating portfolio composition over time...")

        # Pivot to get value_usd per token per date
        value_pivot = portfolio_df.pivot(index="date", columns="symbol", values="value_usd").fillna(0)
        price_pivot = portfolio_df.pivot(index="date", columns="symbol", values="price_usd").fillna(0)

        # Total portfolio value per day
        value_pivot["total"] = value_pivot.sum(axis=1)

        # Composition (weight %) per token
        composition_df = value_pivot.drop(columns=["total"]).div(value_pivot["total"], axis=0)
        composition_df.columns = [f"{col}_weight" for col in composition_df.columns]

        print("\n=== Portfolio Composition (Weights) ===")
        print(composition_df.tail(10))

        # 8. Calculate daily returns per token from prices
        print("\n[*] Calculating daily returns...")

        # Log returns from prices
        log_returns = np.log(price_pivot / price_pivot.shift(1)).fillna(0)
        log_returns.columns = [f"{col}_return" for col in log_returns.columns]

        print("\n=== Daily Log Returns ===")
        print(log_returns.tail(10))

        # 9. Calculate weighted portfolio return
        print("\n[*] Calculating weighted portfolio returns...")

        # Align composition and returns
        weights = composition_df.values[:, :]  # shape: (days, tokens)
        returns = log_returns.values[:, :]     # shape: (days, tokens)

        # Weighted return = sum of (weight * return) per day
        portfolio_daily_returns = (weights * returns).sum(axis=1)
        portfolio_returns_series = pd.Series(portfolio_daily_returns, index=composition_df.index, name="portfolio_return")

        # 10. Calculate cumulative return
        cumulative_return = (np.exp(portfolio_returns_series.cumsum()) - 1) * 100  # as percentage
        cumulative_return.name = "cumulative_return_pct"

        # Combine into returns DataFrame
        returns_df = pd.DataFrame({
            "date": composition_df.index,
            "portfolio_daily_return": portfolio_daily_returns,
            "cumulative_return_pct": cumulative_return.values
        })
        returns_df["total_value_usd"] = value_pivot["total"].values

        print("\n=== Portfolio Returns ===")
        print(returns_df.tail(20))

        # 11. Run comprehensive performance analysis
        print("\n[*] Running comprehensive performance analysis...")

        # Import analyzer
        import sys
        sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.macrocrypto.analytics import WalletPerformanceAnalyzer

        analyzer = WalletPerformanceAnalyzer(portfolio_df)

        # Fetch BTC prices for benchmark
        print("    Fetching BTC benchmark prices...")
        btc_prices = get_historical_prices(
            "0x0000000000000000000000000000000000000000",  # Use ETH endpoint with BTC symbol
            start_time,
            end_time
        )

        # If no BTC prices, fetch via symbol
        if not btc_prices:
            btc_response = requests.post(
                f"https://api.g.alchemy.com/prices/v1/{ALCHEMY_API_KEY}/tokens/historical",
                json={
                    "symbol": "BTC",
                    "startTime": start_time,
                    "endTime": end_time,
                    "interval": "1d"
                },
                headers={"Content-Type": "application/json"}
            )
            if btc_response.status_code == 200:
                result = btc_response.json()
                data = result.get("data", result)
                price_list = data.get("prices", []) if isinstance(data, dict) else data
                for p in price_list:
                    ts = p.get("timestamp")
                    val = p.get("value")
                    if ts and val:
                        try:
                            date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
                            btc_prices[date] = float(val)
                        except:
                            pass

        # Build BTC returns series
        benchmark_returns = None
        if btc_prices:
            btc_series = pd.Series(btc_prices).sort_index()
            btc_series.index = pd.to_datetime(btc_series.index)
            benchmark_returns = np.log(btc_series / btc_series.shift(1)).dropna()
            print(f"    BTC benchmark: {len(benchmark_returns)} days")

        # Get all metrics
        metrics = analyzer.get_all_metrics(benchmark_returns)

        print(f"\n{'='*50}")
        print("       PORTFOLIO PERFORMANCE REPORT")
        print(f"{'='*50}")
        print(f"\n--- Return Metrics ---")
        print(f"Total Return:     {metrics['total_return']*100:>10.2f}%")
        print(f"CAGR:             {metrics['cagr']*100:>10.2f}%")

        print(f"\n--- Risk-Adjusted Metrics ---")
        print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:    {metrics['sortino_ratio']:>10.2f}")
        print(f"Calmar Ratio:     {metrics['calmar_ratio']:>10.2f}")

        print(f"\n--- Risk Metrics ---")
        print(f"Volatility (Ann): {metrics['volatility']*100:>10.2f}%")
        print(f"VaR (95%):        {metrics['var_95']*100:>10.2f}%")
        print(f"CVaR (95%):       {metrics['cvar_95']*100:>10.2f}%")
        print(f"Max Drawdown:     {metrics['max_drawdown']*100:>10.2f}%")

        if 'beta' in metrics:
            print(f"\n--- Relative to BTC ---")
            print(f"Beta:             {metrics['beta']:>10.2f}")
            print(f"Alpha (Ann):      {metrics['alpha']*100:>10.2f}%")
            print(f"Treynor Ratio:    {metrics['treynor_ratio']:>10.2f}")
            print(f"Information Ratio:{metrics['information_ratio']:>10.2f}")

        print(f"\n--- Portfolio Info ---")
        print(f"Start Value:      ${metrics['start_value']:>10,.2f}")
        print(f"End Value:        ${metrics['end_value']:>10,.2f}")
        print(f"Period:           {metrics['days']:>10} days")
        print(f"{'='*50}")

        # Save composition and returns
        composition_df.to_csv("portfolio_composition.csv")
        returns_df.to_csv("portfolio_returns.csv", index=False)

        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv("portfolio_metrics.csv", index=False)

        # Save to CSV
        portfolio_df.to_csv("portfolio_timeseries.csv", index=False)
        total_value.to_csv("portfolio_total_value.csv", index=False)
        df.to_csv("token_transfers.csv", index=False)

        print("\n[*] Saved:")
        print("    - token_transfers.csv (raw transfers)")
        print("    - portfolio_timeseries.csv (per-token daily values)")
        print("    - portfolio_total_value.csv (total portfolio value)")
        print("    - portfolio_composition.csv (daily weights per token)")
        print("    - portfolio_returns.csv (daily & cumulative returns)")
        print("    - portfolio_metrics.csv (performance metrics)")
    else:
        print("[!] No portfolio data to display")


if __name__ == "__main__":
    main()
