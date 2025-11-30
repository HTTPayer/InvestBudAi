// src/server.ts
import express from "express";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { registerInvestBudTools } from "./tools/investbudTools.js";
import { registerInvestBudPrompts } from "./tools/investbudPrompts.js";
import { stopPaymentBridge } from "./services/paymentBridge.js";

// Crear MCP server
const server = new McpServer(
  {
    name: "investbud-mcp",
    version: "0.1.0",
    title: "InvestBud Toolbox"
  },
  {
    debouncedNotificationMethods: [
      "notifications/tools/list_changed"
    ]
  }
);

// Registrar tools y prompts
registerInvestBudTools(server);
registerInvestBudPrompts(server);

// Detectar modo de ejecución
const isStdioMode = process.argv.includes("--stdio") || !process.env.PORT;

if (isStdioMode) {
  // Modo stdio para Claude Desktop (sin console.log)
  const transport = new StdioServerTransport();
  
  // Cleanup al cerrar
  process.on('SIGINT', () => {
    stopPaymentBridge();
    process.exit(0);
  });
  
  process.on('SIGTERM', () => {
    stopPaymentBridge();
    process.exit(0);
  });
  
  server.connect(transport).catch((err) => {
    process.stderr.write(`Failed to start MCP server: ${err}\n`);
    process.exit(1);
  });
} else {
  // Modo HTTP para uso remoto
  const app = express();
  app.use(express.json());

  app.get("/", (req, res) => {
    res.json({
      status: "ok",
      server: "investbud-mcp",
      version: "0.1.0",
      endpoints: {
        mcp: "/mcp (POST only)"
      },
      message: "Server is running. Use POST /mcp for MCP requests."
    });
  });

  app.get("/mcp", (req, res) => {
    res.status(405).json({
      error: "Method Not Allowed",
      message: "The /mcp endpoint only accepts POST requests with MCP protocol.",
      hint: "This is an MCP server endpoint. Use an MCP client to connect."
    });
  });

  app.post("/mcp", async (req, res) => {
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true
    });

    res.on("close", () => {
      transport.close();
    });

    try {
      await server.connect(transport);
      await transport.handleRequest(req, res, req.body);
    } catch (err) {
      if (!res.headersSent) {
        res.status(500).json({ error: "Internal MCP server error" });
      }
    }
  });

  app.use((req, res) => {
    res.status(404).json({
      error: "Not Found",
      message: `Route ${req.originalUrl} not found`,
      availableEndpoints: {
        healthCheck: "GET /",
        mcp: "POST /mcp"
      }
    });
  });

  const PORT = parseInt(process.env.PORT || "3030", 10);
  app.listen(PORT, () => {
    console.log(`🚀 InvestBud MCP server running on http://localhost:${PORT}`);
    console.log(`   Health check: http://localhost:${PORT}/`);
    console.log(`   MCP endpoint: http://localhost:${PORT}/mcp (POST only)`);
  });
}
