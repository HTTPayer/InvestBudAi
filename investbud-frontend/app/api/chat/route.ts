import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 
  'https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Get X-PAYMENT header if present
    const paymentHeader = request.headers.get('x-payment');
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    
    // Forward X-PAYMENT header to backend
    if (paymentHeader) {
      headers['X-PAYMENT'] = paymentHeader;
    }

    const response = await fetch(`${BACKEND_URL}/chat`, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    });

    const data = await response.json();

    // Forward the response with the same status code
    return NextResponse.json(data, { 
      status: response.status,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, X-PAYMENT',
      }
    });
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to connect to backend' },
      { status: 500 }
    );
  }
}

export async function OPTIONS(request: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-PAYMENT',
    },
  });
}
