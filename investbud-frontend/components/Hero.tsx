'use client';

import { TrendingUp, Shield } from 'lucide-react';

export default function Hero() {
  const scrollToChat = () => {
    document.getElementById('chat-section')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section id="hero" className="relative bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white overflow-hidden pt-16">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSAxMCAwIEwgMCAwIDAgMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjAzKSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')] opacity-40"></div>

      <div className="relative max-w-7xl mx-auto px-6 py-24 lg:py-32">
        <div className="text-center max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/20 rounded-full text-sm text-blue-400 mb-8">
            <Shield className="w-4 h-4" />
            Not financial advice · Research & education tool
          </div>

          <h1 className="text-5xl lg:text-7xl font-bold mb-6 leading-tight">
            Your AI Co-Pilot for{' '}
            <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Crypto Investing
            </span>
          </h1>

          <p className="text-xl lg:text-2xl text-gray-300 mb-12 leading-relaxed">
            Real-time macro regime signals + on-chain portfolio analysis,
            <br />
            powered by LLM intelligence.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={scrollToChat}
              className="px-8 py-4 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 shadow-lg shadow-blue-500/30"
            >
              <TrendingUp className="w-5 h-5" />
              Start Analyzing
            </button>
            <button className="px-8 py-4 bg-white/10 hover:bg-white/20 backdrop-blur-sm text-white rounded-lg font-semibold transition-colors border border-white/20">
              View MCP Tools
            </button>
          </div>

          <div className="mt-16 grid grid-cols-3 gap-8 max-w-2xl mx-auto">
            <div>
              <div className="text-3xl font-bold text-blue-400 mb-2">Real-Time</div>
              <div className="text-sm text-gray-400">Macro Signals</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-cyan-400 mb-2">On-Chain</div>
              <div className="text-sm text-gray-400">Portfolio Analysis</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-green-400 mb-2">AI-Powered</div>
              <div className="text-sm text-gray-400">Insights</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
