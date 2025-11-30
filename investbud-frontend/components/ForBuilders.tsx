import { Code2, Zap, GitBranch } from 'lucide-react';

export default function ForBuilders() {
  return (
    <section id="for-builders" className="py-24 bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <div>
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/20 rounded-full text-sm text-blue-400 mb-6">
              <Code2 className="w-4 h-4" />
              For Developers
            </div>
            <h2 className="text-4xl lg:text-5xl font-bold mb-6">
              Built for Integration
            </h2>
            <p className="text-xl text-gray-300 mb-8 leading-relaxed">
              Investbud AI exposes MCP (Model Context Protocol) tools for seamless
              integration into your workflows and applications.
            </p>

            <div className="space-y-4 mb-8">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-blue-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                  <Zap className="w-4 h-4 text-blue-400" />
                </div>
                <div>
                  <h4 className="font-semibold text-white mb-1">LLM Workflows</h4>
                  <p className="text-gray-400">
                    Use with Claude, GPT-4, or any MCP-compatible AI system
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-cyan-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                  <GitBranch className="w-4 h-4 text-cyan-400" />
                </div>
                <div>
                  <h4 className="font-semibold text-white mb-1">
                    Automation Platforms
                  </h4>
                  <p className="text-gray-400">
                    Integrate with n8n, Make, Zapier, and custom pipelines
                  </p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                  <Code2 className="w-4 h-4 text-green-400" />
                </div>
                <div>
                  <h4 className="font-semibold text-white mb-1">Custom DeFi Apps</h4>
                  <p className="text-gray-400">
                    Access regime signals and portfolio analysis as callable functions
                  </p>
                </div>
              </div>
            </div>

            <button className="px-8 py-4 bg-white text-gray-900 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
              View MCP Documentation
            </button>
          </div>

          <div className="bg-gray-800 rounded-2xl p-8 border border-gray-700 shadow-2xl">
            <div className="flex items-center gap-2 mb-4">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
              </div>
              <span className="text-sm text-gray-400 ml-2">mcp-config.json</span>
            </div>
            <pre className="text-sm text-gray-300 overflow-x-auto">
              <code>{`{
  "mcpServers": {
    "investbud": {
      "url": "https://api.investbud.ai",
      "tools": [
        {
          "name": "get_regime",
          "description": "Get macro regime signal"
        },
        {
          "name": "analyze_portfolio",
          "description": "Analyze wallet holdings"
        },
        {
          "name": "get_advisory",
          "description": "Get investment insights"
        }
      ]
    }
  }
}`}</code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}
