import { Icon } from '../ui/Icon';
import { IntroAcousticGraphic } from './IntroAcousticGraphic';

export function IntroSection({ scrollToDemo }: { scrollToDemo: () => void }) {
  return (
    <section className="intro-section" id="top">
      <div className="intro-content">
        <p className="intro-kicker">AI Voice Detection</p>
        <h1 className="intro-heading">
          An <span className="intro-ear-highlight">ear</span><br />trained to notice.
        </h1>
        <p className="intro-summary">
          Vaak analyzes how a voice sounds across time, looking for acoustic evidence of synthetic speech.
        </p>
        <div className="intro-actions">
          <button className="hero-cta-btn" onClick={scrollToDemo} aria-label="Let Vaak listen">
            <span>LET VAAK LISTEN</span>
            <span className="hero-cta-arrow" aria-hidden="true">
              <Icon name="arrow" />
            </span>
          </button>
          <span className="hero-cta-caption">
            <span className="hero-cta-dot" /> 01 / LISTEN
          </span>
        </div>
      </div>
      <div className="intro-orange-circle" aria-hidden="true">
        <IntroAcousticGraphic />
      </div>
    </section>
  );
}
