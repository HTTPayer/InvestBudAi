#!/usr/bin/env node
// test-full-flow.js
// Comprehensive test for the full payment flow with investbudClient

import { analyzeWallet, getMacroRegime } from './dist/services/investbudClient.js';

async function testFullFlow() {
  console.log('🧪 Testing Full InvestBud Payment Flow with Payment Bridge\n');
  console.log('This will:');
  console.log('1. Make a request to InvestBud API');
  console.log('2. Receive 402 Payment Required');
  console.log('3. Open browser for MetaMask payment');
  console.log('4. Retry with X-PAYMENT header');
  console.log('5. Return successful response\n');

  try {
    console.log('📊 Test 1: Getting macro regime (should work without payment or minimal payment)...\n');
    
    const regime = await getMacroRegime();
    console.log('\n✅ Regime response:', JSON.stringify(regime, null, 2));

  } catch (error) {
    console.error('\n❌ Test 1 failed:', error.message);
    console.error('Stack:', error.stack);
  }

  try {
    console.log('\n\n📈 Test 2: Analyzing wallet (requires payment)...');
    console.log('Wallet: 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045');
    console.log('Network: base-sepolia\n');
    
    const portfolio = await analyzeWallet({
      network: 'base-sepolia',
      address: '0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045'
    });
    
    console.log('\n✅ Portfolio analysis successful!');
    console.log('Response:', JSON.stringify(portfolio, null, 2));
    
    console.log('\n🎉 All tests passed!');
    process.exit(0);

  } catch (error) {
    console.error('\n❌ Test 2 failed:', error.message);
    console.error('Stack:', error.stack);
    
    if (error.response) {
      console.error('Response status:', error.response.status);
      console.error('Response data:', error.response.data);
    }
    
    process.exit(1);
  }
}

console.log('════════════════════════════════════════════════════════');
console.log('  InvestBud Payment Bridge - Full Integration Test');
console.log('════════════════════════════════════════════════════════\n');

testFullFlow();
