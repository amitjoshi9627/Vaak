export function PrinciplesSection() {
  const principles = [
    {
      n: "01",
      tag: "UNCERTAINTY BOUNDS",
      title: "Uncertainty stays visible.",
      copy: "Results distinguish raw acoustic model scores, probability estimates, and uncertainty bounds instead of collapsing evidence into false certainty."
    },
    {
      n: "02",
      tag: "TRACEABILITY",
      title: "Evidence has a location.",
      copy: "Temporal analysis must show precisely where and when a signal changed, without inventing human explanations the model cannot substantiate."
    },
    {
      n: "03",
      tag: "IN-MEMORY PRIVACY",
      title: "Private by intent.",
      copy: "Submitted audio is processed in memory and discarded after analysis. It is not retained, indexed, or used for model training."
    }
  ];

  return (
    <section className="principles-section" id="principles">
      {/* Visual Narrative Connector */}
      <div className="section-narrative-cue principles-cue" aria-hidden="true">
        <span className="narrative-line" />
        <span className="narrative-node">06 / RESPONSIBLE INTERPRETATION</span>
        <span className="narrative-line" />
      </div>

      <div className="principles-container">
        {/* Centered Section Intro with Precision Symmetry */}
        <div className="section-intro principles-intro centered-intro">
          <h2>Trust is not a<br />confidence score.</h2>
          <p className="principles-lead-copy">
            Speech analysis should never present statistical estimations as absolute truth. We engineer Vaak around three core principles that ensure scientific rigor, spatial traceability, and strict privacy.
          </p>
        </div>

        {/* Cohesive 3-Column Pillar Cards */}
        <div className="principles-grid">
          {principles.map(p => (
            <article key={p.n} className="principle-card">
              <div className="principle-card-top">
                <span className="principle-num">{p.n}</span>
                <span className="principle-tag">{p.tag}</span>
              </div>
              <h3>{p.title}</h3>
              <p>{p.copy}</p>
            </article>
          ))}
        </div>

        {/* Properly Aligned Forensic Notice */}
        <div className="principles-status-bar">
          <div className="status-badge-wrap">
            <span className="status-badge">INTERPRETATION STANDARD</span>
          </div>
          <p>
            Vaak provides evidence to support human review. Its output is an analytical signal, not an autonomous final verdict.
          </p>
        </div>
      </div>
    </section>
  );
}
