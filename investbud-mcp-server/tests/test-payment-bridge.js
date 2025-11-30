#!/usr/bin/env node
// test-payment-bridge.js
// Script para probar el Payment Bridge sin necesidad de Claude Desktop

import { getPaymentBridge } from './dist/services/paymentBridge.js';

async function testPaymentBridge() {
  console.log('🧪 Testing Payment Bridge...\n');

  try {
    // Crear requisitos de pago de prueba
    const testRequirements = {
      x402Version: 1,
      accepts: [{
        scheme: 'exact',
        network: 'base-sepolia',
        payTo: '0x1d4ba461fdba577dfe0400252cdf462e5a1ff13f',
        asset: '0x036CbD53842c5426634e7929541eC2318f3dCF7e',
        maxAmountRequired: '50000',
        description: 'Test payment for InvestBud API',
        mimeType: 'application/json',
        maxTimeoutSeconds: 180
      }]
    };

    console.log('📋 Payment Requirements:');
    console.log(JSON.stringify(testRequirements, null, 2));
    console.log('\n🌐 Starting payment bridge server...');

    // Obtener el bridge (esto inicia el servidor)
    const bridge = await getPaymentBridge();
    console.log('✅ Bridge server started\n');

    console.log('🚀 Opening browser for payment...');
    console.log('📝 Please approve the payment in your browser\n');

    // Solicitar pago (esto abrirá el navegador)
    const paymentHeader = await bridge.requestPayment(testRequirements);

    console.log('\n✅ Payment completed successfully!');
    console.log('🔐 Payment Header:', paymentHeader.substring(0, 50) + '...');
    
    process.exit(0);
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    process.exit(1);
  }
}

testPaymentBridge();
