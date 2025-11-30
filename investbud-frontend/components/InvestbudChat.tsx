'use client';

import { useState, useRef } from 'react';
import { Send, TrendingUp, TrendingDown, Wallet } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import type { ChatMessage, RegimeSignal, PortfolioSummary } from '../types/chat';
import { useMetaMask } from '../hooks/useMetaMask';
import { sendMessageWithX402 } from '../lib/x402';
import WalletModal from './WalletModal';

export default function InvestbudChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(() => crypto.randomUUID());
  const [regimeSignal, setRegimeSignal] = useState<RegimeSignal>();
  const [paymentStatus, setPaymentStatus] = useState<string>('');
  const [showWalletModal, setShowWalletModal] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const { address, isConnecting, error: walletError, provider, connect, disconnect, isConnected } = useMetaMask();

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    // Check wallet connection - show modal if not connected
    if (!isConnected || !provider) {
      setShowWalletModal(true);
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setPaymentStatus('');

    try {
      const metadata = {
        user_agent: navigator.userAgent,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        wallet_address: address,
      };

      // x402-fetch handles the entire payment flow automatically
      setPaymentStatus('💳 Processing payment...');
      
      const data = await sendMessageWithX402(
        provider,
        sessionId,
        userMessage.content,
        metadata
      );

      if (data.error) {
        throw new Error(data.error);
      }

      if (data.regime_signal) {
        setRegimeSignal(data.regime_signal as unknown as RegimeSignal);
      }

      // Parse the backend response - it comes as a Python dict string
      let backendResponseText = '';
      
      if (data.response) {
        try {
          // Convert Python dict string to JSON
          const pythonDictStr = data.response
            .replace(/'/g, '"')  // Replace single quotes with double quotes
            .replace(/True/g, 'true')
            .replace(/False/g, 'false')
            .replace(/None/g, 'null');
          
          const parsed = JSON.parse(pythonDictStr);
          backendResponseText = parsed.response || data.response;
        } catch {
          // Fallback to raw response if parsing fails
          backendResponseText = data.response;
        }
      } else if (data.reply) {
        backendResponseText = data.reply;
      } else {
        backendResponseText = JSON.stringify(data);
      }

      // Process response through RAG for compliance and context
      setPaymentStatus('🤖 Processing with AI...');
      
      const ragResponse = await fetch('/api/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          userMessage: userMessage.content,
          backendResponse: backendResponseText,
          userContext: JSON.stringify({
            regime: data.regime_signal,
            portfolio: data.portfolio_summary,
            wallet_address: address,
          }),
        }),
      });

      if (!ragResponse.ok) {
        throw new Error('Failed to process with AI');
      }

      const { enhancedResponse } = await ragResponse.json();

      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: enhancedResponse,
        timestamp: new Date(),
        regime_signal: data.regime_signal as RegimeSignal | undefined,
        portfolio_summary: data.portfolio_summary as PortfolioSummary | undefined,
      };

      setMessages(prev => [...prev, assistantMessage]);
      setPaymentStatus('✅ Complete');
      
      // Clear status after 2 seconds
      setTimeout(() => setPaymentStatus(''), 2000);

    } catch (error) {
      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
      setPaymentStatus('❌ Transaction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleConnectWallet = async () => {
    const success = await connect();
    if (success) {
      setShowWalletModal(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      <WalletModal
        isOpen={showWalletModal}
        onClose={() => setShowWalletModal(false)}
        onConnect={handleConnectWallet}
        isConnecting={isConnecting}
        error={walletError}
      />
      
      <div className="flex flex-col h-[600px] bg-white rounded-lg shadow-xl border border-gray-200">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-linear-to-r from-gray-50 to-white">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-linear-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center text-white font-bold">
              IB
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">Investbud AI</h3>
              {regimeSignal && (
                <div className="flex items-center gap-1.5 text-xs">
                  {regimeSignal.current === 'risk-on' ? (
                    <TrendingUp className="w-3 h-3 text-green-600" />
                  ) : (
                    <TrendingDown className="w-3 h-3 text-red-600" />
                  )}
                  <span className="font-medium capitalize">{regimeSignal.current}</span>
                  <span className="text-gray-400">·</span>
                  <span className="text-gray-500">
                    {new Date(regimeSignal.last_updated).toLocaleTimeString()}
                  </span>
                </div>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {isConnected ? (
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-2 px-3 py-1.5 bg-green-50 border border-green-200 rounded-lg">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  <span className="text-xs font-medium text-green-700">
                    {address?.slice(0, 6)}...{address?.slice(-4)}
                  </span>
                </div>
                <button
                  onClick={disconnect}
                  className="text-xs text-gray-600 hover:text-gray-900 px-2 py-1 rounded hover:bg-gray-100"
                >
                  Disconnect
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowWalletModal(true)}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                <Wallet className="w-4 h-4" />
                Connect Wallet
              </button>
            )}
          </div>
        </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {!isConnected && (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-linear-to-br from-blue-500 to-cyan-500 rounded-2xl mx-auto mb-4 flex items-center justify-center text-white text-2xl font-bold">
              <Wallet className="w-8 h-8" />
            </div>
            <h4 className="text-lg font-semibold text-gray-900 mb-2">
              Connect Your Wallet
            </h4>
            <p className="text-sm text-gray-600 mb-4 max-w-sm mx-auto">
              Connect your wallet to start chatting with Investbud AI.
            </p>
            <button
              onClick={() => setShowWalletModal(true)}
              className="px-6 py-3 bg-linear-to-r from-blue-600 to-cyan-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-cyan-700 transition-all shadow-lg"
            >
              Connect Wallet
            </button>
          </div>
        )}

        {isConnected && messages.length === 0 && (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-linear-to-br from-blue-500 to-cyan-500 rounded-2xl mx-auto mb-4 flex items-center justify-center text-white text-2xl font-bold">
              💬
            </div>
            <h4 className="text-lg font-semibold text-gray-900 mb-2">
              Start a Conversation
            </h4>
            <p className="text-sm text-gray-600 mb-4 max-w-sm mx-auto">
              Ask about macro regimes, portfolio analysis, or market conditions.
            </p>
            <div className="flex flex-col gap-2 text-xs text-gray-500">
              <p>&quot;What&apos;s the current macro regime?&quot;</p>
              <p>&quot;Analyze wallet 0x123...&quot;</p>
              <p>&quot;How correlated is BTC to risk assets?&quot;</p>
            </div>
          </div>
        )}

        {isConnected && messages.length > 0 && (
          <>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-3 ${
                    message.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  {message.role === 'user' ? (
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  ) : (
                    <div className="text-sm markdown-content">
                      <ReactMarkdown
                        components={{
                          h1: (props) => <h1 className="text-lg font-bold mb-2 mt-3" {...props} />,
                          h2: (props) => <h2 className="text-base font-bold mb-2 mt-3" {...props} />,
                          h3: (props) => <h3 className="text-sm font-semibold mb-1 mt-2" {...props} />,
                          p: (props) => <p className="mb-2 leading-relaxed" {...props} />,
                          ul: (props) => <ul className="list-disc list-inside mb-2 space-y-1" {...props} />,
                          ol: (props) => <ol className="list-decimal list-inside mb-2 space-y-1" {...props} />,
                          li: (props) => <li className="ml-2" {...props} />,
                          strong: (props) => <strong className="font-semibold" {...props} />,
                          em: (props) => <em className="italic" {...props} />,
                          code: (props) => <code className="bg-gray-200 px-1 py-0.5 rounded text-xs" {...props} />,
                          blockquote: (props) => <blockquote className="border-l-4 border-gray-300 pl-3 italic my-2" {...props} />,
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>
                  )}

                  {message.portfolio_summary && (
                    <div className="mt-3 pt-3 border-t border-gray-300">
                      <p className="text-xs font-semibold mb-1">Portfolio Summary</p>
                      <p className="text-xs">
                        Total: ${message.portfolio_summary.total_value_usd.toLocaleString()}
                      </p>
                      <div className="mt-1 text-xs">
                        {message.portfolio_summary.top_holdings.slice(0, 3).map((holding: { symbol: string; percentage: number }) => (
                          <div key={holding.symbol}>
                            {holding.symbol}: {holding.percentage.toFixed(1)}%
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start mb-4">
                <div className="max-w-[80%] px-4 py-3 rounded-2xl bg-gradient-to-r from-gray-100 to-gray-50 text-gray-800 shadow-sm border border-gray-200">
                  <div className="flex items-center space-x-3">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '0ms'}}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></div>
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></div>
                    </div>
                    <span className="text-sm font-medium bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                      Thinking...
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </>
        )}
      </div>

        <div className="border-t border-gray-200 p-4 bg-gray-50">
          {paymentStatus && (
            <div className="mb-3 px-3 py-2 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-700 text-center">
              {paymentStatus}
            </div>
          )}
          
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyPress}
              placeholder={isConnected ? "Type your question..." : "Connect wallet to start..."}
              className="flex-1 px-4 py-2 border text-black border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={isLoading || !isConnected}
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !input.trim() || !isConnected}
              aria-label="Send message"
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          
          <p className="text-xs text-gray-500 mt-2 text-center">
            💳 Each message requires USDC payment · You&apos;ll sign with MetaMask · ⓘ Not financial advice
          </p>
        </div>
      </div>
    </>
  );
}
