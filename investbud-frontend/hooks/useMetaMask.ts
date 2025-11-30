'use client';

import { useState, useEffect } from 'react';
import { ethers } from 'ethers';
import { BASE_SEPOLIA_CHAIN_ID, BASE_SEPOLIA_RPC } from '../lib/constants';

export function useMetaMask() {
  const [address, setAddress] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [provider, setProvider] = useState<ethers.BrowserProvider | null>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && window.ethereum) {
      const provider = new ethers.BrowserProvider(window.ethereum);
      setProvider(provider);

      // Listen for account changes
      window.ethereum.on('accountsChanged', (...args: unknown[]) => {
        const accounts = args[0] as string[];
        if (accounts.length === 0) {
          setAddress(null);
        } else {
          setAddress(accounts[0]);
        }
      });

      // Listen for chain changes - DON'T reload, just update the provider
      window.ethereum.on('chainChanged', () => {
        const newProvider = new ethers.BrowserProvider(window.ethereum!);
        setProvider(newProvider);
      });

      // DON'T auto-connect - only connect when user clicks button
    }

    return () => {
      if (window.ethereum) {
        // Clean up event listeners (note: can't remove all, just stop adding new ones)
      }
    };
  }, []);

  const connect = async () => {
    if (!window.ethereum) {
      setError('MetaMask is not installed');
      return false;
    }

    try {
      setIsConnecting(true);
      setError(null);

      // Request account access
      const result = await window.ethereum.request({ 
        method: 'eth_requestAccounts' 
      });
      const accounts = result as string[];

      if (accounts.length > 0) {
        setAddress(accounts[0]);

        // Check and switch to Base Sepolia
        const chainIdResult = await window.ethereum.request({ method: 'eth_chainId' });
        const chainId = chainIdResult as string;
        const currentChainId = parseInt(chainId, 16);

        if (currentChainId !== BASE_SEPOLIA_CHAIN_ID) {
          try {
            await window.ethereum.request({
              method: 'wallet_switchEthereumChain',
              params: [{ chainId: `0x${BASE_SEPOLIA_CHAIN_ID.toString(16)}` }],
            });
          } catch (switchError: unknown) {
            // Chain doesn't exist, add it
            if (switchError && typeof switchError === 'object' && 'code' in switchError && switchError.code === 4902) {
              await window.ethereum.request({
                method: 'wallet_addEthereumChain',
                params: [{
                  chainId: `0x${BASE_SEPOLIA_CHAIN_ID.toString(16)}`,
                  chainName: 'Base Sepolia',
                  nativeCurrency: {
                    name: 'Ethereum',
                    symbol: 'ETH',
                    decimals: 18,
                  },
                  rpcUrls: [BASE_SEPOLIA_RPC],
                  blockExplorerUrls: ['https://sepolia.basescan.org'],
                }],
              });
            } else {
              throw switchError;
            }
          }
        }
        
        return true; // Connection successful
      }
      
      return false;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect');
      return false;
    } finally {
      setIsConnecting(false);
    }
  };

  const disconnect = () => {
    setAddress(null);
  };

  return {
    address,
    isConnecting,
    error,
    provider,
    connect,
    disconnect,
    isConnected: !!address,
  };
}

declare global {
  interface Window {
    ethereum?: {
      request: (args: { method: string; params?: unknown[] }) => Promise<unknown>;
      on: (event: string, handler: (...args: unknown[]) => void) => void;
      removeListener: (event: string, handler: (...args: unknown[]) => void) => void;
      isMetaMask?: boolean;
    };
  }
}
