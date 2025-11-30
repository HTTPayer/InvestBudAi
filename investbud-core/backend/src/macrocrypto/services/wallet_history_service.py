"""
Wallet History Service

Fetches and constructs historical portfolio data from on-chain transfers.
"""

import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from ..analytics import WalletPerformanceAnalyzer
from ..config import get_transfer_categories, supports_internal_transfers


class WalletHistoryService:
    """Service for fetching and analyzing wallet historical data."""

    def __init__(self, alchemy_api_key: str, network: str = "eth-mainnet"):
        self.api_key = alchemy_api_key
        self.network = network
        self.base_url = f"https://{network}.g.alchemy.com/v2/{alchemy_api_key}"
        self.prices_url = f"https://api.g.alchemy.com/prices/v1/{alchemy_api_key}/tokens/historical"

    def _hex_to_int(self, hex_str: str) -> Optional[int]:
        """Convert hex string to int."""
        if hex_str is None:
            return None
        try:
            return int(hex_str, 16)
        except (ValueError, TypeError):
            return None

    def _get_asset_transfers(self, address: str, direction: str = "to") -> List[dict]:
        """Fetch all asset transfers for an address with pagination."""
        all_transfers = []
        page_key = None

        # Use network-specific transfer categories (internal only on eth/polygon)
        categories = get_transfer_categories(self.network)

        while True:
            params = {
                "fromBlock": "0x0",
                "toBlock": "latest",
                "category": categories,
                "withMetadata": True,
                "maxCount": "0x3e8",
                "excludeZeroValue": False,  # Include zero-value transfers (gas-only txs)
            }

            if direction == "to":
                params["toAddress"] = address
            else:
                params["fromAddress"] = address

            if page_key:
                params["pageKey"] = page_key

            response = requests.post(
                self.base_url,
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
            all_transfers.extend(transfers)

            page_key = result["result"].get("pageKey")
            if not page_key:
                break

        return all_transfers

    def _parse_transfers(self, transfers: List[dict], direction: str, seen_ids: set) -> List[dict]:
        """Parse transfers into rows with sign based on direction."""
        rows = []

        for transfer in transfers:
            unique_id = transfer.get("uniqueId")
            if unique_id in seen_ids:
                continue
            seen_ids.add(unique_id)

            block_number = self._hex_to_int(transfer.get("blockNum"))
            value = transfer.get("value") or 0

            metadata = transfer.get("metadata") or {}
            block_timestamp = metadata.get("blockTimestamp")
            if block_timestamp:
                timestamp = datetime.fromisoformat(block_timestamp.replace("Z", "+00:00"))
            else:
                timestamp = None

            signed_value = value if direction == "to" else -value

            raw_contract = transfer.get("rawContract") or {}
            token_address = raw_contract.get("address")
            token_decimals = self._hex_to_int(raw_contract.get("decimal"))

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

    def _get_current_eth_balance(self, address: str) -> float:
        """Get current ETH balance from the chain using eth_getBalance."""
        response = requests.post(
            self.base_url,
            json={
                "jsonrpc": "2.0",
                "method": "eth_getBalance",
                "params": [address, "latest"],
                "id": 1
            }
        )
        result = response.json()
        if "result" in result:
            balance_wei = int(result["result"], 16)
            return balance_wei / 1e18
        return 0.0

    def _get_eth_balances_at_blocks(self, address: str, blocks: List[int]) -> Dict[int, float]:
        """
        Get ETH balances at multiple blocks using batch requests.

        This fetches the actual on-chain ETH balance at each block, which already
        accounts for gas spent up to that point. More accurate than cumulative
        transfer calculations.

        Args:
            address: Wallet address
            blocks: List of block numbers (integers)

        Returns:
            Dict mapping block_number -> balance_eth
        """
        balances = {}
        batch_size = 100

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

            response = requests.post(self.base_url, json=batch_request)

            if response.status_code == 200:
                results = response.json()
                for idx, result in enumerate(results):
                    if "result" in result:
                        block_num = batch[idx]
                        balance_wei = int(result["result"], 16)
                        balances[block_num] = balance_wei / 1e18

        return balances

    def _calculate_gas_from_balance(self, address: str, calculated_eth: float) -> Tuple[float, float]:
        """
        Calculate gas spent by comparing calculated ETH balance with actual on-chain balance.

        This is a fallback method - prefer _get_gas_costs_from_receipts for accuracy.

        Returns:
            Tuple of (actual_balance, gas_spent)
        """
        actual_balance = self._get_current_eth_balance(address)
        gas_spent = calculated_eth - actual_balance
        # Gas can't be negative (would indicate missed transfers, not negative gas)
        gas_spent = max(0, gas_spent)
        return actual_balance, gas_spent

    def _get_transaction_receipts_batch(self, tx_hashes: List[str]) -> List[dict]:
        """
        Fetch transaction receipts in batches using JSON-RPC batch requests.

        Args:
            tx_hashes: List of transaction hashes

        Returns:
            List of receipt objects
        """
        receipts = []
        batch_size = 100  # Alchemy supports up to 1000, but 100 is safer

        for i in range(0, len(tx_hashes), batch_size):
            batch = tx_hashes[i:i + batch_size]

            # Build batch request
            batch_request = [
                {
                    "jsonrpc": "2.0",
                    "method": "eth_getTransactionReceipt",
                    "params": [tx_hash],
                    "id": idx
                }
                for idx, tx_hash in enumerate(batch)
            ]

            response = requests.post(self.base_url, json=batch_request)

            if response.status_code == 200:
                results = response.json()
                for result in results:
                    if "result" in result and result["result"]:
                        receipts.append(result["result"])

        return receipts

    def _get_gas_costs_from_receipts(
        self,
        transfers_df: pd.DataFrame,
        wallet_address: str
    ) -> Tuple[pd.DataFrame, float]:
        """
        Get per-transaction gas costs by fetching receipts.

        Only counts gas for transactions where wallet is the TRANSACTION SENDER
        (the one who signed and paid gas), not just where wallet appears in transfers.

        Args:
            transfers_df: DataFrame with transfers (must have tx_hash, from_address, timestamp, date)
            wallet_address: The wallet address (lowercase)

        Returns:
            Tuple of (gas_df with columns [date, timestamp, tx_hash, gas_eth], total_gas_spent)
        """
        wallet_address = wallet_address.lower()

        # Get ALL unique tx hashes (we need to check who actually paid gas)
        all_tx_hashes = transfers_df["tx_hash"].unique()

        if len(all_tx_hashes) == 0:
            return pd.DataFrame(columns=["date", "timestamp", "tx_hash", "gas_eth"]), 0.0

        # Fetch receipts
        receipts = self._get_transaction_receipts_batch(list(all_tx_hashes))

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

            gas_used = self._hex_to_int(receipt.get("gasUsed")) or 0
            effective_gas_price = self._hex_to_int(receipt.get("effectiveGasPrice")) or 0

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

    def _get_historical_prices(self, token_address: str, start_time: str, end_time: str) -> Dict:
        """Fetch historical prices for a token. Handles 365-day API limit by chunking."""
        prices = {}

        # Parse dates
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

        # Chunk into 364-day periods (API limit is 365)
        max_days = 364
        current_start = start_dt

        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=max_days), end_dt)

            chunk_start = current_start.isoformat().replace("+00:00", "Z")
            chunk_end = current_end.isoformat().replace("+00:00", "Z")

            if token_address == "0x0000000000000000000000000000000000000000":
                payload = {
                    "symbol": "ETH",
                    "startTime": chunk_start,
                    "endTime": chunk_end,
                    "interval": "1d"
                }
            else:
                payload = {
                    "network": self.network,
                    "address": token_address,
                    "startTime": chunk_start,
                    "endTime": chunk_end,
                    "interval": "1d"
                }

            response = requests.post(self.prices_url, json=payload, headers={"Content-Type": "application/json"})

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

            current_start = current_end + timedelta(days=1)

        return prices

    def _get_btc_prices(self, start_time: str, end_time: str) -> Dict:
        """Fetch BTC historical prices for benchmark. Handles 365-day API limit by chunking."""
        prices = {}

        # Parse dates
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

        # Chunk into 364-day periods
        max_days = 364
        current_start = start_dt

        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=max_days), end_dt)

            chunk_start = current_start.isoformat().replace("+00:00", "Z")
            chunk_end = current_end.isoformat().replace("+00:00", "Z")

            response = requests.post(
                self.prices_url,
                json={
                    "symbol": "BTC",
                    "startTime": chunk_start,
                    "endTime": chunk_end,
                    "interval": "1d"
                },
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                data = result.get("data", result)
                price_list = data.get("prices", []) if isinstance(data, dict) else data
                for p in price_list:
                    ts = p.get("timestamp")
                    val = p.get("value")
                    if ts and val:
                        try:
                            date = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
                            prices[date] = float(val)
                        except:
                            pass

            current_start = current_end + timedelta(days=1)

        return prices

    def get_wallet_history(self, wallet_address: str) -> Dict:
        """
        Get complete wallet history with transfers, balances, and prices.

        Returns:
            Dict with transfers_df, daily_balances, portfolio_df, composition_df, returns_df
        """
        wallet_address = wallet_address.lower()

        # 1. Fetch transfers
        inflows = self._get_asset_transfers(wallet_address, direction="to")
        outflows = self._get_asset_transfers(wallet_address, direction="from")

        # 2. Parse and deduplicate
        seen_ids = set()
        rows = []
        rows.extend(self._parse_transfers(inflows, "to", seen_ids))
        rows.extend(self._parse_transfers(outflows, "from", seen_ids))

        if not rows:
            return {"error": "No transfers found for this wallet"}

        df = pd.DataFrame(rows)

        # Filter spam tokens
        spam_indicators = ["claim", "reward", "visit", "airdrop", ".org", ".com", ".io"]
        df = df[~df["symbol"].str.lower().str.contains("|".join(spam_indicators), na=False)]

        if df.empty:
            return {"error": "No valid transfers after filtering"}

        eth_address = "0x0000000000000000000000000000000000000000"

        # 3. Calculate daily balances from transfers (cumulative) - matches test9's calculate_daily_balances()
        daily = df.groupby(["date", "token_address", "symbol"]).agg({
            "value": "sum",
            "token_decimals": "first"
        }).reset_index()
        daily = daily.sort_values(["token_address", "date"])
        daily["balance"] = daily.groupby("token_address")["value"].cumsum()

        # 4. For ETH: replace cumulative balances with actual on-chain balances at each transfer block
        # This accounts for gas spent automatically (same logic as test9)
        gas_spent = 0.0
        eth_transfers = df[df["token_address"] == eth_address].copy()

        if not eth_transfers.empty:
            # Get unique blocks where ETH was transferred
            eth_blocks = eth_transfers["block_number"].dropna().unique().astype(int).tolist()
            eth_blocks = sorted(set(eth_blocks))
            print(f"[DEBUG] Found {len(eth_blocks)} ETH transfer blocks")

            # Fetch actual balances at each block
            block_balances = self._get_eth_balances_at_blocks(wallet_address, eth_blocks)
            print(f"[DEBUG] Got {len(block_balances)} block balances")
            if block_balances:
                last_block = max(block_balances.keys())
                print(f"[DEBUG] Last block {last_block} balance: {block_balances[last_block]:.8f} ETH")

            if block_balances:
                # Create mapping: block -> date
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

                    # Calculate gas spent (difference between transfer-calculated and actual)
                    # Recalculate from original df like test9 does
                    calculated_eth_df = daily[daily["token_address"] == eth_address]
                    if not calculated_eth_df.empty:
                        calculated_final = calculated_eth_df["balance"].iloc[-1]
                        actual_final = eth_df["balance"].iloc[-1]
                        gas_spent = max(0, calculated_final - actual_final)

                    # Replace ETH entries in daily_balances with actual balances
                    daily = daily[daily["token_address"] != eth_address]
                    daily = pd.concat([daily, eth_df], ignore_index=True)
                    daily = daily.sort_values(["token_address", "date"])
                    print(f"[DEBUG] After replacement - final ETH balance: {actual_final:.8f} ETH")
                    print(f"[DEBUG] Gas spent (calculated - actual): {gas_spent:.8f} ETH")

        # 6. Get date range and fetch prices
        tokens = df[["token_address", "symbol"]].drop_duplicates()
        min_date = df["date"].min()
        max_date = max(df["date"].max(), datetime.now().date())

        start_time = datetime.combine(min_date, datetime.min.time()).isoformat() + "Z"
        end_time = datetime.combine(max_date, datetime.max.time()).isoformat() + "Z"

        price_data = {}
        for _, token in tokens.iterrows():
            token_addr = token["token_address"]
            if token_addr:
                price_data[token_addr] = self._get_historical_prices(token_addr, start_time, end_time)

        # 7. Build portfolio timeseries
        date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        portfolio_rows = []

        for token_addr in tokens["token_address"].unique():
            if not token_addr:
                continue

            token_df = daily[daily["token_address"] == token_addr].copy()
            if token_df.empty:
                continue

            symbol = token_df["symbol"].iloc[0]
            prices = price_data.get(token_addr, {})

            token_ts = pd.DataFrame({"date": date_range.date})
            token_ts = token_ts.merge(token_df[["date", "balance"]], on="date", how="left")
            token_ts["balance"] = token_ts["balance"].ffill().fillna(0)
            token_ts["price_usd"] = token_ts["date"].map(prices).fillna(0)
            token_ts["value_usd"] = token_ts["balance"] * token_ts["price_usd"]
            token_ts["token_address"] = token_addr
            token_ts["symbol"] = symbol

            portfolio_rows.append(token_ts)

        if not portfolio_rows:
            return {"error": "Could not build portfolio timeseries"}

        portfolio_df = pd.concat(portfolio_rows, ignore_index=True)

        # 7. Filter tokens with no value
        tokens_with_value = portfolio_df.groupby("token_address")["value_usd"].sum()
        tokens_to_keep = tokens_with_value[tokens_with_value > 0].index
        portfolio_df = portfolio_df[portfolio_df["token_address"].isin(tokens_to_keep)]

        if portfolio_df.empty:
            return {"error": "No tokens with USD value"}

        # 7b. Filter out leading dates with negative/zero total value
        daily_totals = portfolio_df.groupby("date")["value_usd"].sum()
        positive_dates = daily_totals[daily_totals > 0].index
        if len(positive_dates) == 0:
            return {"error": "No dates with positive portfolio value"}
        first_positive_date = positive_dates.min()
        portfolio_df = portfolio_df[portfolio_df["date"] >= first_positive_date]
        min_date = first_positive_date  # Update min_date for return value

        # 8. Calculate composition and returns
        value_pivot = portfolio_df.pivot(index="date", columns="symbol", values="value_usd").fillna(0)
        price_pivot = portfolio_df.pivot(index="date", columns="symbol", values="price_usd").fillna(0)

        value_pivot["total"] = value_pivot.sum(axis=1)
        composition_df = value_pivot.drop(columns=["total"]).div(value_pivot["total"], axis=0)
        composition_df.columns = [f"{col}_weight" for col in composition_df.columns]

        # Calculate log returns, handling inf/nan from zero prices
        price_ratio = price_pivot / price_pivot.shift(1)
        price_ratio = price_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        log_returns = np.log(price_ratio).fillna(0)
        log_returns = log_returns.replace([np.inf, -np.inf], 0)
        log_returns.columns = [f"{col}_return" for col in log_returns.columns]

        weights = composition_df.values
        returns = log_returns.values
        portfolio_daily_returns = (weights * returns).sum(axis=1)
        # Handle any remaining inf/nan
        portfolio_daily_returns = np.nan_to_num(portfolio_daily_returns, nan=0.0, posinf=0.0, neginf=0.0)
        cumulative_return = (np.exp(np.cumsum(portfolio_daily_returns)) - 1) * 100

        returns_df = pd.DataFrame({
            "date": composition_df.index,
            "portfolio_daily_return": portfolio_daily_returns,
            "cumulative_return_pct": cumulative_return,
            "total_value_usd": value_pivot["total"].values
        })

        # 9. Get BTC for benchmark
        btc_prices = self._get_btc_prices(start_time, end_time)
        benchmark_returns = None
        if btc_prices:
            btc_series = pd.Series(btc_prices).sort_index()
            btc_series.index = pd.to_datetime(btc_series.index)
            benchmark_returns = np.log(btc_series / btc_series.shift(1)).dropna()

        return {
            "transfers_df": df,
            "daily_balances": daily,
            "portfolio_df": portfolio_df,
            "composition_df": composition_df,
            "returns_df": returns_df,
            "benchmark_returns": benchmark_returns,
            "gas_spent": gas_spent,
            "start_date": str(min_date),
            "end_date": str(max_date),
        }

    def get_performance_metrics(self, wallet_address: str) -> Dict:
        """
        Get comprehensive performance metrics for a wallet.

        Returns:
            Dict with all performance metrics
        """
        history = self.get_wallet_history(wallet_address)

        if "error" in history:
            return history

        portfolio_df = history["portfolio_df"]
        benchmark_returns = history["benchmark_returns"]

        # Run analyzer
        analyzer = WalletPerformanceAnalyzer(portfolio_df)
        metrics = analyzer.get_all_metrics(benchmark_returns)

        # Add gas info
        metrics["gas_spent_eth"] = history["gas_spent"]

        return metrics

    def get_historical_data(self, wallet_address: str) -> Dict:
        """
        Get historical data for a wallet (for /wallet/historical endpoint).

        Returns:
            Dict with returns, composition, daily balances
        """
        history = self.get_wallet_history(wallet_address)

        if "error" in history:
            return history

        # Convert DataFrames to JSON-serializable format
        returns_df = history["returns_df"]
        composition_df = history["composition_df"]
        portfolio_df = history["portfolio_df"]

        # Daily balances with value_usd per token
        daily_data = portfolio_df.groupby("date").apply(
            lambda x: {
                "total_value_usd": x["value_usd"].sum(),
                "tokens": x[["symbol", "balance", "price_usd", "value_usd"]].to_dict("records")
            }
        ).to_dict()

        return {
            "returns": returns_df.to_dict("records"),
            "composition": {
                "dates": [str(d) for d in composition_df.index],
                "weights": composition_df.to_dict("list")
            },
            "daily_balances": {
                str(k): v for k, v in daily_data.items()
            },
            "start_date": history["start_date"],
            "end_date": history["end_date"],
        }

    def get_composition_history(self, wallet_address: str) -> Dict:
        """
        Get portfolio composition (weights) over time.

        Returns:
            Dict with dates, composition by token, and total_value_usd
        """
        history = self.get_wallet_history(wallet_address)

        if "error" in history:
            return history

        portfolio_df = history["portfolio_df"]

        # Build analyzer to get composition
        analyzer = WalletPerformanceAnalyzer(portfolio_df)

        # Get weights timeseries
        weights = analyzer.weights
        total_value = analyzer.total_value

        # Format for API response
        dates = [str(d) for d in weights.index]

        composition = {}
        for symbol in weights.columns:
            composition[symbol] = [
                round(float(v), 6) if not pd.isna(v) else 0.0
                for v in weights[symbol].values
            ]

        total_value_usd = [
            round(float(v), 2) if not pd.isna(v) else 0.0
            for v in total_value.values
        ]

        return {
            "wallet_address": wallet_address.lower(),
            "dates": dates,
            "composition": composition,
            "total_value_usd": total_value_usd,
            "start_date": history["start_date"],
            "end_date": history["end_date"],
        }

    def get_rolling_metrics(
        self,
        wallet_address: str,
        window: int = 30,
        metrics: List[str] = None
    ) -> Dict:
        """
        Get rolling window metrics for time-series visualization.

        Args:
            wallet_address: Wallet address to analyze
            window: Rolling window in days (default 30)
            metrics: List of metrics to calculate ['sharpe', 'sortino', 'volatility', 'beta']

        Returns:
            Dict with dates and rolling metric arrays
        """
        if metrics is None:
            metrics = ['sharpe', 'sortino', 'volatility']

        history = self.get_wallet_history(wallet_address)

        if "error" in history:
            return history

        portfolio_df = history["portfolio_df"]
        benchmark_returns = history.get("benchmark_returns")

        # Build analyzer
        analyzer = WalletPerformanceAnalyzer(portfolio_df)

        # Get rolling metrics
        rolling_data = analyzer.get_rolling_metrics(
            window=window,
            metrics=metrics,
            benchmark_returns=benchmark_returns
        )

        return {
            "wallet_address": wallet_address.lower(),
            "window": window,
            "metrics_calculated": metrics,
            **rolling_data,
            "start_date": history["start_date"],
            "end_date": history["end_date"],
        }
