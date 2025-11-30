'use client';

import { MessageSquare } from 'lucide-react';
import InvestbudChat from './InvestbudChat';

export default function ChatSection() {
  return (
    <section id="chat-section" className="py-24 bg-white">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-50 border border-blue-200 rounded-full text-sm text-blue-600 mb-6">
              <MessageSquare className="w-4 h-4" />
              For Investors
            </div>
            <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-6">
              Chat-First Analysis
            </h2>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed">
              Talk to Investbud AI like you&apos;d talk to an analyst. Get instant insights
              on macro conditions, portfolio composition, and market context.
            </p>

            <div className="space-y-3 text-gray-600">
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span>&quot;What&apos;s the current macro regime?&quot;</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-cyan-500 rounded-full"></div>
                <span>&quot;Analyze wallet 0x123...&quot;</span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>&quot;How correlated is my portfolio to BTC?&quot;</span>
              </div>
            </div>

            <div className="mt-8 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <p className="text-sm text-gray-700">
                <span className="font-semibold">Powered by RAG:</span> Retrieval-augmented
                generation with our research docs, plus live calls to macro and on-chain
                data sources.
              </p>
            </div>
          </div>

          <div>
            <InvestbudChat />
          </div>
        </div>
      </div>
    </section>
  );
}
