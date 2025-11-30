// src/tools/investbudPrompts.ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

export function registerInvestBudPrompts(server: McpServer) {
  // 1) Market Analysis Prompt
  server.registerPrompt(
    "market_analysis",
    {
      title: "Comprehensive Market Analysis",
      description: "Get a complete market analysis including macro regime and smart money flows."
    },
    async () => {
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Please perform a comprehensive market analysis by:
1. First, get the current macro regime using get_macro_regime
2. Then, get the latest smart money flow analysis using get_news
3. Synthesize both analyses to provide:
   - Current market conditions (Risk-On/Risk-Off)
   - Key tokens with smart money accumulation
   - Market sentiment and outlook
   - Trading opportunities based on both signals

Provide a clear, actionable summary for investors.`
            }
          }
        ]
      };
    }
  );

  // 2) Wallet Deep Dive Prompt
  server.registerPrompt(
    "wallet_deep_dive",
    {
      title: "Deep Dive Wallet Analysis",
      description: "Comprehensive analysis of a crypto wallet with investment recommendations.",
      argsSchema: {
        wallet_address: z.string().describe("The wallet address to analyze"),
        network: z.string().optional().describe("Network (e.g., 'eth-mainnet', 'base-mainnet', 'polygon-mainnet')")
      }
    },
    async (args) => {
      const address = args.wallet_address as string;
      const network = (args.network as string) || "base-mainnet";
      
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Please perform a deep dive analysis of wallet ${address} on ${network}:

1. Analyze the wallet composition using analyze_wallet
2. Get the current macro regime using get_macro_regime
3. Get portfolio advice using advise_portfolio with medium risk preference
4. Check latest smart money flows using get_news

Provide a comprehensive report including:
- Current portfolio composition and risk metrics
- Alignment with macro conditions
- Comparison with smart money flows
- Specific recommendations for portfolio optimization
- Risk-adjusted action items`
            }
          }
        ]
      };
    }
  );

  // 3) Smart Money Tracker Prompt
  server.registerPrompt(
    "smart_money_tracker",
    {
      title: "Smart Money Flow Tracker",
      description: "Track and analyze smart money movements in the crypto market."
    },
    async () => {
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Please track smart money flows by:

1. Get the latest smart money analysis using get_news
2. Get the current macro regime using get_macro_regime

Provide a report on:
- Top tokens by smart money netflow (accumulation vs distribution)
- How these flows align with the current macro regime
- Key opportunities where smart money is moving
- Tokens to watch in the next 24-48 hours
- Risk assessment for following these flows

Format the response with clear token recommendations and rationale.`
            }
          }
        ]
      };
    }
  );

  // 4) Oracle Signal Publisher Prompt
  server.registerPrompt(
    "publish_oracle_signal",
    {
      title: "Publish Macro Signal to Oracle",
      description: "Publish the current macro regime signal to the on-chain oracle.",
      argsSchema: {
        network: z.string().optional().describe("Network to publish to (e.g., 'base-sepolia', 'base-mainnet')")
      }
    },
    async (args) => {
      const network = (args.network as string) || "base-sepolia";
      
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Please publish the current macro signal to the on-chain oracle:

1. First, get the current macro regime using get_macro_regime to see what will be published
2. Then, submit the signal to the oracle on ${network} using submit_signal
3. Verify the transaction and provide the details

Explain:
- What signal is being published (Risk-On/Risk-Off and probability)
- The transaction details and status
- How this signal can be used by on-chain protocols
- Next update schedule`
            }
          }
        ]
      };
    }
  );

  // 5) Portfolio Rebalancing Prompt
  server.registerPrompt(
    "portfolio_rebalance",
    {
      title: "Portfolio Rebalancing Strategy",
      description: "Get rebalancing recommendations based on market conditions and wallet analysis.",
      argsSchema: {
        wallet_address: z.string().describe("The wallet address to rebalance"),
        network: z.string().optional().describe("Network (e.g., 'eth-mainnet', 'base-mainnet')"),
        risk_preference: z.string().optional().describe("Risk preference: low, medium, or high")
      }
    },
    async (args) => {
      const address = args.wallet_address as string;
      const network = (args.network as string) || "base-mainnet";
      const risk = (args.risk_preference as string) || "medium";
      
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Please create a portfolio rebalancing strategy for wallet ${address} on ${network}:

1. Analyze current portfolio using analyze_wallet
2. Get macro regime using get_macro_regime
3. Check smart money flows using get_news
4. Get personalized advice using advise_portfolio with ${risk} risk preference

Provide a detailed rebalancing plan:
- Current allocation vs recommended allocation
- Tokens to buy/sell with specific percentages
- Timing recommendations based on macro regime
- Risk management considerations
- Expected outcomes and timeline

Make the recommendations actionable and specific.`
            }
          }
        ]
      };
    }
  );

  // 6) Market Regime Change Alert
  server.registerPrompt(
    "regime_change_alert",
    {
      title: "Market Regime Change Strategy",
      description: "Get investment strategy adjustments when market regime changes."
    },
    async () => {
      return {
        messages: [
          {
            role: "user",
            content: {
              type: "text",
              text: `Analyze the current market regime and provide regime change strategies:

1. Get current macro regime using get_macro_regime
2. Get smart money flows using get_news

Based on the current regime, provide:
- Is the market in Risk-On or Risk-Off mode?
- Probability and confidence level
- Key indicators supporting this regime
- Historical context (how long in this regime?)
- What assets perform best in this regime
- Early warning signs of regime change
- Recommended portfolio positioning
- Specific action items for the next 7 days

Include both offensive (growth) and defensive strategies based on regime probability.`
            }
          }
        ]
      };
    }
  );
}
