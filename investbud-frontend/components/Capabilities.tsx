import { Gauge, PieChart, TrendingUp, LineChart } from 'lucide-react';

export default function Capabilities() {
  const capabilities = [
    {
      icon: Gauge,
      title: 'Macro Regime Classification',
      description:
        'Risk-On / Risk-Off signals derived from FRED macro data and crypto market structure. Know the macro backdrop before you position.',
      gradient: 'from-blue-500 to-cyan-500',
    },
    {
      icon: PieChart,
      title: 'Portfolio Analysis',
      description:
        'On-chain wallet breakdown: allocation, diversification, concentration risk. Understand what you actually own.',
      gradient: 'from-cyan-500 to-teal-500',
    },
    {
      icon: LineChart,
      title: 'Performance Metrics',
      description:
        'Sharpe ratio, max drawdown, CAGR, beta vs BTC. Quantify your historical performance.',
      gradient: 'from-teal-500 to-green-500',
    },
    {
      icon: TrendingUp,
      title: 'AI Advisory',
      description:
        'Conversational insights combining macro conditions, your portfolio, and market context. Ask questions, get analysis—not orders.',
      gradient: 'from-green-500 to-emerald-500',
    },
  ];

  return (
    <section id="capabilities" className="py-24 bg-gradient-to-br from-gray-50 to-white">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
            Core Capabilities
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Four pillars of intelligent crypto analysis
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {capabilities.map((capability, index) => (
            <div
              key={index}
              className="group bg-white rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 border border-gray-100 hover:border-gray-200"
            >
              <div
                className={`w-16 h-16 mb-6 bg-gradient-to-br ${capability.gradient} rounded-xl flex items-center justify-center shadow-lg group-hover:scale-110 transition-transform duration-300`}
              >
                <capability.icon className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-3">
                {capability.title}
              </h3>
              <p className="text-gray-600 leading-relaxed">
                {capability.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
