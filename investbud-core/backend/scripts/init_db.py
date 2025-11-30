"""
Initialize PostgreSQL database for MacroCrypto chat sessions.

Usage:
    uv run python init_db.py

Prerequisites:
    1. PostgreSQL installed and running
    2. Database created (e.g., CREATE DATABASE macrocrypto;)
    3. DATABASE_URL set in .env file

Example DATABASE_URL formats:
    postgresql://localhost/macrocrypto
    postgresql://user:password@localhost:5432/macrocrypto
    postgresql://user:password@localhost/macrocrypto?sslmode=require
"""
from dotenv import load_dotenv
import os
from src.macrocrypto.db import init_db, get_session_maker, ChatSession

load_dotenv()

def main():
    """Initialize database tables."""
    try:
        database_url = os.getenv('DATABASE_URL')

        if not database_url:
            print("[!] DATABASE_URL not set in .env file")
            print("\nPlease add to .env:")
            print("DATABASE_URL=postgresql://localhost/macrocrypto")
            return

        print(f"Initializing database: {database_url}")

        # Create tables
        engine = init_db(database_url)
        print("[OK] Database tables created successfully")

        # Test connection
        SessionMaker = get_session_maker(database_url)
        session = SessionMaker()

        # Check if we can query
        count = session.query(ChatSession).count()
        print(f"[OK] Database connection verified - {count} chat sessions found")

        session.close()

        print("\n[OK] Database initialization complete!")
        print("\nYou can now start the API server:")
        print("  uv run python start_api.py")

    except Exception as e:
        print(f"[ERROR] Database initialization failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure PostgreSQL is installed and running")
        print("2. Create the database: psql -c 'CREATE DATABASE macrocrypto;'")
        print("3. Check DATABASE_URL format in .env")
        print("4. Verify PostgreSQL user has appropriate permissions")

if __name__ == "__main__":
    main()
