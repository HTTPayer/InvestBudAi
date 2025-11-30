#!/usr/bin/env node

/**
 * Test script para debuggear el payment bridge
 * Abre el navegador y muestra dónde encontrar los logs
 */

import { getPaymentBridge, stopPaymentBridge } from './dist/services/paymentBridge.js';

console.log('\n════════════════════════════════════════════════════════');
console.log('  🐛 Payment Bridge Debug Test');
console.log('════════════════════════════════════════════════════════\n');

console.log('📋 INSTRUCCIONES:');
console.log('1. Se abrirá tu navegador en http://localhost:3402-3502');
console.log('2. Abre las DevTools (F12 o Cmd+Opt+I)');
console.log('3. Ve a la pestaña "Console"');
console.log('4. Conecta MetaMask cuando se solicite');
console.log('5. Firma la transacción');
console.log('6. COPIA TODOS los logs que empiecen con 🔍 DEBUG');
console.log('7. Pégalos aquí en el chat\n');

console.log('⏳ Iniciando en 3 segundos...\n');

await new Promise(resolve => setTimeout(resolve, 3000));

try {
  // Simular 402 Payment Required response
  const mockRequirements = {
    x402Version: 1,
    accepts: [{
      scheme: 'exact',
      network: 'base-sepolia',
      maxAmountRequired: '10000',
      resource: 'http://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com/regime',
      payTo: '0x1d4ba461fdba577dfe0400252cdf462e5a1ff13f',
      maxTimeoutSeconds: 120,
      asset: '0x036CbD53842c5426634e7929541eC2318f3dCF7e',
      extra: {
        name: 'USDC',
        version: '2'
      }
    }]
  };

  console.log('🔄 Iniciando payment bridge...');
  const bridge = await getPaymentBridge();
  
  console.log('🔄 Solicitando pago...');
  console.log('💡 RECUERDA: Mira la consola del NAVEGADOR, no esta terminal\n');
  
  const paymentHeader = await bridge.requestPayment(mockRequirements);
  
  console.log('\n✅ Payment header obtenido:');
  console.log(paymentHeader.substring(0, 100) + '...\n');
  
  console.log('📝 Ahora copia TODOS los logs de la consola del navegador');
  console.log('   que empiecen con 🔍 DEBUG y pégalos en el chat.\n');
  
  stopPaymentBridge();
  process.exit(0);
  
} catch (error) {
  console.error('\n❌ Error:', error.message);
  console.error('\n💡 Si viste logs en el navegador, cópialos de todos modos\n');
  stopPaymentBridge();
  process.exit(1);
}
