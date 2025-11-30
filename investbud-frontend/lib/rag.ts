'use client';

/**
 * Get user context from browser cookies and localStorage
 */
export function getUserContext(): string {
  if (typeof window === 'undefined') return '';

  const context: string[] = [];

  // Get cookies
  const cookies = document.cookie.split(';');
  cookies.forEach(cookie => {
    const [key, value] = cookie.split('=').map(s => s.trim());
    if (key && value) {
      context.push(`Cookie ${key}: ${value}`);
    }
  });

  // Get localStorage data
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key) {
        const value = localStorage.getItem(key);
        if (value) {
          context.push(`LocalStorage ${key}: ${value}`);
        }
      }
    }
  } catch {
    // Silently fail if localStorage is not available
  }

  // Get session history if available
  const sessionHistory = sessionStorage.getItem('investbud_history');
  if (sessionHistory) {
    context.push(`Previous interactions: ${sessionHistory}`);
  }

  return context.join('\n');
}

/**
 * Store interaction in session for context building
 */
export function storeInteraction(message: string, response: string) {
  if (typeof window === 'undefined') return;

  try {
    const history = JSON.parse(sessionStorage.getItem('investbud_history') || '[]');
    history.push({
      timestamp: new Date().toISOString(),
      message,
      response,
    });

    // Keep only last 10 interactions
    const recentHistory = history.slice(-10);
    sessionStorage.setItem('investbud_history', JSON.stringify(recentHistory));
  } catch {
    // Silently fail if sessionStorage is not available
  }
}

/**
 * Process message through RAG and LLM via API route (secure)
 */
export async function processWithRAG(
  userMessage: string,
  backendResponse: string
): Promise<string> {
  const userContext = getUserContext();

  try {
    const response = await fetch('/api/rag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        userMessage,
        backendResponse,
        userContext,
      }),
    });

    const data = await response.json();

    if (data.success) {
      const enhancedResponse = data.response;
      // Store this interaction for future context
      storeInteraction(userMessage, enhancedResponse);
      return enhancedResponse;
    } else {
      return backendResponse;
    }
  } catch {
    // Fallback to backend response if RAG fails
    return backendResponse;
  }
}
