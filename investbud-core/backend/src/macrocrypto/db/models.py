"""
Database models for MacroCrypto.
"""
from sqlalchemy import Column, String, Text, DateTime, Integer, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()


class ChatSession(Base):
    """Chat session model for storing conversation history."""

    __tablename__ = 'chat_sessions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    messages = Column(Text, nullable=False, default='[]')  # JSON array of messages
    wallet_address = Column(String(42), nullable=True)
    network = Column(String(50), nullable=True, default='eth-mainnet')
    portfolio_snapshot = Column(Text, nullable=True)  # Cached wallet analysis (JSON)
    portfolio_updated_at = Column(DateTime, nullable=True)  # Last portfolio fetch
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<ChatSession(session_id='{self.session_id}', messages_count={len(self.get_messages())})>"

    def get_messages(self):
        """Parse messages JSON."""
        import json
        return json.loads(self.messages) if self.messages else []

    def set_messages(self, messages_list):
        """Set messages as JSON."""
        import json
        self.messages = json.dumps(messages_list)

    def get_portfolio_snapshot(self):
        """Parse portfolio snapshot JSON."""
        import json
        return json.loads(self.portfolio_snapshot) if self.portfolio_snapshot else None

    def set_portfolio_snapshot(self, portfolio_data):
        """Set portfolio snapshot as JSON."""
        import json
        self.portfolio_snapshot = json.dumps(portfolio_data) if portfolio_data else None
        self.portfolio_updated_at = datetime.utcnow()


# Database configuration
def get_database_url():
    """Get database URL from environment variables."""
    return os.getenv(
        'DATABASE_URL'
    )


def init_db(database_url=None):
    """Initialize database and create tables."""
    url = database_url or get_database_url()
    engine = create_engine(url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session_maker(database_url=None):
    """Get SQLAlchemy session maker."""
    url = database_url or get_database_url()
    engine = create_engine(url, echo=False, pool_pre_ping=True)
    return sessionmaker(bind=engine)
