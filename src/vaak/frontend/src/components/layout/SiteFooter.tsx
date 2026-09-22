import { Icon } from '../ui/Icon';
import { scrollToListen } from '../../utils/scroll';

export function SiteFooter() {
  const scrollToTop = (e: React.MouseEvent) => {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const scrollToDemo = (e: React.MouseEvent) => {
    e.preventDefault();
    scrollToListen();
  };

  return (
    <footer className="site-footer">
      <div className="footer-container">
        {/* Main Grid: 4 distinct columns */}
        <div className="footer-grid">
          {/* Col 1: Brand & Identity */}
          <div className="footer-brand-col">
            <div className="footer-accent-rule" aria-hidden="true" />
            <div className="footer-brand">
              <span className="footer-wordmark">VAAK</span>
              <span className="footer-sep" aria-hidden="true" />
              <span className="footer-sanskrit" lang="hi">वाक्</span>
              <span className="footer-dot" aria-hidden="true">·</span>
            </div>
            <p className="footer-tagline">An ear trained to notice.</p>
            <p className="footer-bio">
              A voice analysis system designed to detect synthetic speech and show where its model finds stronger evidence across time.
            </p>
            <div className="footer-status-pill">
              <span className="footer-status-dot" />
              <span>AI Voice Detection · Public Preview</span>
            </div>
          </div>

          {/* Col 2: Navigation / Explore */}
          <div className="footer-nav-col">
            <h4 className="footer-col-header">Explore</h4>
            <ul className="footer-links">
              <li>
                <a href="#listen" onClick={scrollToDemo} className="footer-link">
                  Listen
                </a>
              </li>
              <li>
                <a href="#evidence" className="footer-link">
                  Evidence
                </a>
              </li>
              <li>
                <a href="#method" className="footer-link">
                  How It Listens
                </a>
              </li>
              <li>
                <a href="#evaluation" className="footer-link">
                  Evaluation
                </a>
              </li>
              <li>
                <a href="#principles" className="footer-link">
                  Principles
                </a>
              </li>
            </ul>
          </div>

          {/* Col 3: Technical Specs */}
          <div className="footer-nav-col">
            <h4 className="footer-col-header">Technical</h4>
            <div className="footer-specs-board" aria-label="System specifications">
              <div className="footer-spec-row">
                <span className="footer-spec-key">INPUT</span>
                <span className="footer-spec-val">16kHz Mono PCM</span>
              </div>
              <div className="footer-spec-row">
                <span className="footer-spec-key">FRAME</span>
                <span className="footer-spec-val">40ms · 20ms Hop</span>
              </div>
              <div className="footer-spec-row">
                <span className="footer-spec-key">FEATURES</span>
                <span className="footer-spec-val">Spectral Residuals</span>
              </div>
              <div className="footer-spec-row">
                <span className="footer-spec-key">OUTPUT</span>
                <span className="footer-spec-val">Likelihood (0.0–1.0)</span>
              </div>
            </div>
          </div>

          {/* Col 4: Trust & Privacy */}
          <div className="footer-privacy-col">
            <h4 className="footer-col-header">Trust & Privacy</h4>
            <div className="footer-privacy-box">
              <div className="footer-privacy-head">
                <Icon name="shield" className="footer-shield-icon" />
                <span>PRIVATE BY DESIGN</span>
              </div>
              <p className="footer-privacy-text">
                Submitted audio is processed in memory and discarded after analysis. It is not retained, indexed, or used for model training.
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Sub-Footer Strip: 3 balanced zones */}
        <div className="footer-bottom-strip">
          <div className="footer-meta-left">
            <span className="footer-copyright">© 2026 VAAK</span>
            <span className="footer-meta-sep">/</span>
            <span className="footer-open-standard">Built for careful listening.</span>
          </div>

          <div className="footer-meta-center">
            <a
              href="https://github.com"
              target="_blank"
              rel="noreferrer"
              className="footer-bottom-github-link"
              aria-label="GitHub Repository"
            >
              <span>GitHub Repository</span>
              <Icon name="arrow" className="footer-github-arrow" />
            </a>
          </div>

          <div className="footer-meta-right">
            <span className="footer-version-tag">Public preview v0.5.0</span>
            <span className="footer-meta-sep" aria-hidden="true">·</span>
            <button className="footer-clean-top-btn" onClick={scrollToTop} aria-label="Back to top of page">
              <span>Back to top</span>
              <Icon name="arrow" className="footer-clean-top-arrow" />
            </button>
          </div>
        </div>
      </div>
    </footer>
  );
}
