import './styles/index.css';
import { useEffect } from 'react';

import { SiteHeader } from './components/layout/SiteHeader';
import { SiteFooter } from './components/layout/SiteFooter';
import { IntroSection } from './components/sections/IntroSection';
import { ListenSection } from './components/analysis/ListenSection';
import { EvidenceSection } from './components/sections/EvidenceSection';
import { MethodSection } from './components/sections/MethodSection';
import { EvaluationSection } from './components/sections/EvaluationSection';
import { PrinciplesSection } from './components/sections/PrinciplesSection';
import { scrollToListen } from './utils/scroll';

export default function App() {
  useEffect(() => {
    if (window.location.hash === '#listen') {
      setTimeout(() => {
        scrollToListen();
      }, 100);
    }
  }, []);

  return (
    <main>
      <SiteHeader scrollToDemo={scrollToListen} />
      <IntroSection scrollToDemo={scrollToListen} />
      <ListenSection />
      <EvidenceSection />
      <MethodSection />
      <EvaluationSection />
      <PrinciplesSection />
      <SiteFooter />
    </main>
  );
}
