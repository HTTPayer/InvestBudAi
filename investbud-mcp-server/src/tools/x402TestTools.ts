// import { Tool } from "@modelcontextprotocol/sdk/types.js";
// import { createWalletClient, http } from "viem";
// import { privateKeyToAccount } from "viem/accounts";
// import { wrapFetchWithPayment } from "x402-fetch";
// import { baseSepolia } from "viem/chains";

// /**
//  * Tool dedicado para probar x402-fetch
//  */
// export const x402TestTool: Tool = {
//   name: "test_x402_fetch",
//   description: "Tool de prueba para verificar que x402-fetch funciona correctamente. Hace una petición a un endpoint de prueba usando x402.",
//   inputSchema: {
//     type: "object",
//     properties: {
//       url: {
//         type: "string",
//         description: "URL del endpoint a probar con x402-fetch",
//       },
//       method: {
//         type: "string",
//         enum: ["GET", "POST", "PUT", "DELETE"],
//         description: "Método HTTP a usar",
//         default: "GET",
//       },
//     },
//     required: ["url"],
//   },
// };

// /**
//  * Handler para el tool de prueba x402
//  */
// export async function handleX402Test(args: any) {
//   const { url, method = "GET" } = args;

//   try {
//     console.log(`[X402 Test] Probando x402-fetch con ${method} ${url}`);

//     // Get private key from environment
//     const privateKey = process.env.WALLET_PRIVATE_KEY;
//     if (!privateKey) {
//       throw new Error("WALLET_PRIVATE_KEY no está definida en el archivo .env");
//     }

//     // Create a wallet client
//     const account = privateKeyToAccount(privateKey as `0x${string}`);
//     const client = createWalletClient({
//       account,
//       transport: http(),
//       chain: baseSepolia,
//     });

//     // Wrap the fetch function with payment handling
//     // const fetchWithPay = wrapFetchWithPayment(fetch, client as unknown as );

//     // Make a request that may require payment
//     // const response = await fetchWithPay(url, {
//     //   method,
//     //   headers: {
//     //     "Content-Type": "application/json",
//     //   },
//     // });

//     const data = await response.text();

//     return {
//       content: [
//         {
//           type: "text",
//           text: JSON.stringify(
//             {
//               success: true,
//               status: response.status,
//               statusText: response.statusText,
//               headers: Object.fromEntries(response.headers.entries()),
//               data: data,
//               message: "x402-fetch funcionó correctamente",
//             },
//             null,
//             2
//           ),
//         },
//       ],
//     };
//   } catch (error: any) {
//     console.error("[X402 Test] Error:", error);
//     return {
//       content: [
//         {
//           type: "text",
//           text: JSON.stringify(
//             {
//               success: false,
//               error: error.message,
//               stack: error.stack,
//             },
//             null,
//             2
//           ),
//         },
//       ],
//       isError: true,
//     };
//   }
// }
