export interface ChatRequest {
  session_id: string;
  message: string;
  wallet_address?: string;
  network?: 'ethereum' | 'base' | 'polygon';
  metadata?: {
    user_agent?: string;
    timezone?: string;
  };
}

export interface RegimeSignal {
  current: 'risk-on' | 'risk-off';
  confidence: number;
  last_updated: string;
}

export interface PortfolioSummary {
  total_value_usd: number;
  top_holdings: Array<{ symbol: string; percentage: number }>;
}

export interface ChatResponse {
  session_id: string;
  reply: string;
  message_count: number;
  context_used: string[];
  regime_signal?: RegimeSignal;
  portfolio_summary?: PortfolioSummary;
  error?: string;
  rate_limit?: {
    remaining: number;
    reset_at: string;
  };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  regime_signal?: RegimeSignal;
  portfolio_summary?: PortfolioSummary;
}
