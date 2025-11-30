// src/tools/investbudTools.ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import {
  getMacroRegime,
  analyzeWallet,
  advisePortfolio,
  getNews,
  submitSignal,
  AnalyzeWalletInput,
  AdvisePortfolioInput,
  SubmitSignalInput
} from "../services/investbudClient.js";

export function registerInvestBudTools(server: McpServer) {
  // 1) get_macro_regime -> GET /regime
  server.registerTool(
    "get_macro_regime",
    {
      title: "Get Macro Regime",
      description:
        "Fetches the current macro regime (Risk-On / Risk-Off) from InvestBud, with probabilities and key features.",
      inputSchema: z.object({})
    },
    async () => {
      const data = await getMacroRegime();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(data, null, 2)
          }
        ],
        structuredContent: data
      };
    }
  );

  // 2) analyze_wallet -> POST /portfolio
  server.registerTool(
    "analyze_wallet",
    {
      title: "Analyze Wallet",
      description:
        "Analyze a wallet's composition and risk metrics on a given network using InvestBud's /portfolio endpoint.",
      inputSchema: z.object({
        network: z
          .string()
          .describe(
            "Network identifier, e.g. 'eth-mainnet', 'polygon-mainnet', 'arb-mainnet'. Check /config for full list."
          ),
        address: z
          .string()
          .describe("Wallet address (EVM compatible, checksum recommended).")
      })
    },
    async (args) => {
      const input: AnalyzeWalletInput = {
        network: args.network,
        address: args.address
      };
      const data = await analyzeWallet(input);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(data, null, 2)
          }
        ],
        structuredContent: data
      };
    }
  );

  // 3) advise_portfolio -> POST /advise
  server.registerTool(
    "advise_portfolio",
    {
      title: "Advise Portfolio",
      description:
        "Full advisory using InvestBud: macro signal + wallet analysis + LLM-powered recommendation.",
      inputSchema: z.object({
        network: z
          .string()
          .describe("Network identifier, e.g. 'eth-mainnet', 'polygon-mainnet'."),
        address: z
          .string()
          .describe("Wallet address to advise on."),
        riskPreference: z
          .enum(["low", "medium", "high"])
          .optional()
          .describe(
            "Optional risk preference for the recommendation. Defaults to medium if omitted."
          ),
        notes: z
          .string()
          .optional()
          .describe("Optional extra context for the advisor (e.g. time horizon).")
      })
    },
    async (args) => {
      const input: AdvisePortfolioInput = {
        network: args.network,
        address: args.address,
        riskPreference: args.riskPreference,
        notes: args.notes
      };

      const data = await advisePortfolio(input);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(data, null, 2)
          }
        ],
        structuredContent: data
      };
    }
  );

  // 4) get_news -> GET /news
  server.registerTool(
    "get_news",
    {
      title: "Get Smart Money Flow News",
      description:
        "Get AI-powered analysis of smart money token flows from Nansen, combined with news and market analysis from Heurist, summarized by GPT-4. Returns top 100 tokens by smart money netflow (24h), accumulation/distribution patterns, and actionable insights. Updated every 6 hours. Cost: $0.10 USDC.",
      inputSchema: z.object({})
    },
    async () => {
      const data = await getNews();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(data, null, 2)
          }
        ],
        structuredContent: data
      };
    }
  );

  // 5) submit_signal -> POST /signal
  server.registerTool(
    "submit_signal",
    {
      title: "Submit Macro Signal to On-Chain Oracle",
      description:
        "Submit the current macro regime signal to an on-chain oracle. Predicts current macro regime, creates and signs an update for the oracle contract, and attempts to submit the transaction. Returns signed data regardless of submission success. Cost: $0.10 USDC.",
      inputSchema: z.object({
        network: z
          .string()
          .describe("Network to submit signal to, e.g. 'base-sepolia', 'base-mainnet'.")
      })
    },
    async (args) => {
      const input: SubmitSignalInput = {
        network: args.network
      };

      const data = await submitSignal(input);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(data, null, 2)
          }
        ],
        structuredContent: data
      };
    }
  );
}
