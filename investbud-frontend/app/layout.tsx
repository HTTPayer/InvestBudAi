import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Investbud AI - Asset Investing Co-Pilot',
  description: 'Your AI Co-Pilot for Crypto Investing - Real-time macro regime signals + on-chain portfolio analysis, powered by LLM intelligence.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
