"""
Migrate chat_sessions table to add new columns.

Adds:
- network (for blockchain network)
- portfolio_snapshot (cached portfolio data)
- portfolio_updated_at (cache timestamp)

Usage:
    uv run python migrate_chat_schema.py
"""
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text

load_dotenv()

def migrate():
    """Add new columns to chat_sessions table."""
    database_url = os.getenv('DATABASE_URL')

    if not database_url:
        print("[!] DATABASE_URL not set in .env")
        return

    print(f"Connecting to: {database_url}")
    engine = create_engine(database_url, echo=False)

    with engine.connect() as conn:
        # Check if columns already exist
        result = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name='chat_sessions'
        """))
        existing_columns = {row[0] for row in result}

        print(f"Existing columns: {existing_columns}")

        # Add network column
        if 'network' not in existing_columns:
            print("[OK] Adding 'network' column...")
            conn.execute(text("""
                ALTER TABLE chat_sessions
                ADD COLUMN network VARCHAR(50) DEFAULT 'eth-mainnet'
            """))
            conn.commit()
            print("[OK] Added 'network' column")
        else:
            print("[SKIP] 'network' column already exists")

        # Add portfolio_snapshot column
        if 'portfolio_snapshot' not in existing_columns:
            print("[OK] Adding 'portfolio_snapshot' column...")
            conn.execute(text("""
                ALTER TABLE chat_sessions
                ADD COLUMN portfolio_snapshot TEXT
            """))
            conn.commit()
            print("[OK] Added 'portfolio_snapshot' column")
        else:
            print("[SKIP] 'portfolio_snapshot' column already exists")

        # Add portfolio_updated_at column
        if 'portfolio_updated_at' not in existing_columns:
            print("[OK] Adding 'portfolio_updated_at' column...")
            conn.execute(text("""
                ALTER TABLE chat_sessions
                ADD COLUMN portfolio_updated_at TIMESTAMP
            """))
            conn.commit()
            print("[OK] Added 'portfolio_updated_at' column")
        else:
            print("[SKIP] 'portfolio_updated_at' column already exists")

    print("\n[OK] Migration complete!")

if __name__ == "__main__":
    migrate()
