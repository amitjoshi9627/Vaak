import { Icon } from '../ui/Icon';

export function SiteHeader({ scrollToDemo }: { scrollToDemo: () => void }) {
  return (
    <header className="site-header">
      <a className="wordmark" href="#top" aria-label="Vaak home">
        <span className="wordmark-latin">VAAK</span>
        <span className="wordmark-sep" aria-hidden="true" />
        <span className="wordmark-devanagari" lang="hi">वाक्</span>
        <span className="wordmark-dot" aria-hidden="true">.</span>
      </a>
      <nav aria-label="Main navigation">
        <a href="#evidence">Evidence</a>
        <a href="#method">How It Listens</a>
        <a href="#evaluation">Evaluation</a>
        <a href="#principles">Principles</a>
      </nav>
      <button className="nav-cta" onClick={scrollToDemo} aria-label="Analyze audio">
        <span className="nav-cta-dot" aria-hidden="true" />
        <span className="nav-cta-label">Analyze audio</span>
        <Icon name="arrow" className="nav-cta-arrow" />
      </button>
    </header>
  );
}

