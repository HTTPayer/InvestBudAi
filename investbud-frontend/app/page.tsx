'use client';

import { useEffect } from 'react';
import Header from '@/components/Header';
import Hero from '@/components/Hero';
import HowItWorks from '@/components/HowItWorks';
import Capabilities from '@/components/Capabilities';
import ChatSection from '@/components/ChatSection';
import ForBuilders from '@/components/ForBuilders';
import Disclaimer from '@/components/Disclaimer';
import Footer from '@/components/Footer';

export default function Home() {
  useEffect(() => {
    // Prevent auto-scroll on page load
    window.history.scrollRestoration = 'manual';
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="min-h-screen bg-white">
      <Header />
      <div id="hero">
        <Hero />
      </div>
      <HowItWorks />
      <Capabilities />
      <ChatSection />
      <ForBuilders />
      <Disclaimer />
      <Footer />
    </div>
  );
}
