"""
Check how many rows have NaN values and when we get complete data.
"""
from src.macrocrypto.data import CombinedDataPipeline
from datetime import datetime, timedelta


def main():
    pipeline = CombinedDataPipeline()

    # Fetch 2 years of data
    start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    df = pipeline.fetch_combined_data(start_date=start_date)

    print("\n" + "=" * 70)
    print("NaN ANALYSIS")
    print("=" * 70)

    # Count NaNs per column
    nan_counts = df.isna().sum().sort_values(ascending=False)
    print("\nColumns with most NaNs:")
    print(nan_counts.head(10))

    # Count rows with ANY NaN
    rows_with_nan = df.isna().any(axis=1).sum()
    total_rows = len(df)
    complete_rows = total_rows - rows_with_nan

    print(f"\n" + "=" * 70)
    print(f"Total rows: {total_rows}")
    print(f"Rows with ANY NaN: {rows_with_nan} ({rows_with_nan/total_rows*100:.1f}%)")
    print(f"Complete rows (no NaN): {complete_rows} ({complete_rows/total_rows*100:.1f}%)")

    # Find first row with no NaNs
    first_complete_idx = df.dropna().index[0] if not df.dropna().empty else None
    if first_complete_idx:
        days_until_complete = (first_complete_idx - df.index[0]).days
        print(f"\nFirst complete row (no NaNs): {first_complete_idx}")
        print(f"Days from start until complete data: {days_until_complete}")

    # Show the columns causing the most NaNs
    print("\n" + "=" * 70)
    print("CULPRIT FEATURES (most NaNs):")
    print("=" * 70)
    for col in nan_counts.head(5).index:
        first_valid = df[col].first_valid_index()
        if first_valid:
            days_until_valid = (first_valid - df.index[0]).days
            print(f"{col}: {nan_counts[col]} NaNs (valid after {days_until_valid} days)")

    # Check specific feature
    print("\n" + "=" * 70)
    print("btc_price_vs_ma_200 detail:")
    print("=" * 70)
    btc_ma_200_col = [c for c in df.columns if 'ma_200' in c.lower() and 'vs' in c.lower()]
    if btc_ma_200_col:
        col = btc_ma_200_col[0]
        print(f"First 10 values of {col}:")
        print(df[col].head(10))
        print(f"\nLast 5 values of {col}:")
        print(df[col].tail(5))


if __name__ == '__main__':
    main()
