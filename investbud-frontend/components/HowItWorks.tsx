import { Wallet, Brain, Target } from 'lucide-react';

export default function HowItWorks() {
  const steps = [
    {
      icon: Wallet,
      title: 'Connect Your Data',
      description:
        'Link your wallet or ask about market conditions. Investbud AI analyzes on-chain holdings and macro signals.',
    },
    {
      icon: Brain,
      title: 'Get Context-Aware Insights',
      description:
        'Our LLM combines macro regime classification (Risk-On/Risk-Off), portfolio metrics, and proprietary research.',
    },
    {
      icon: Target,
      title: 'Make Informed Decisions',
      description:
        'Receive clear analysis and educational insights. No black-box recommendations, just transparent data.',
    },
  ];

  return (
    <section id="how-it-works" className="py-24 bg-white">
      <div className="max-w-7xl mx-auto px-6">
        <div className="text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
            How It Works
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Three steps to better crypto investment decisions
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-12">
          {steps.map((step, index) => (
            <div key={index} className="relative">
              {index < steps.length - 1 && (
                <div className="hidden md:block absolute top-12 left-[60%] w-[80%] h-0.5 bg-gradient-to-r from-blue-300 to-transparent"></div>
              )}
              <div className="relative">
                <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/30">
                  <step.icon className="w-12 h-12 text-white" />
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
                  {step.title}
                </h3>
                <p className="text-gray-600 text-center leading-relaxed">
                  {step.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
