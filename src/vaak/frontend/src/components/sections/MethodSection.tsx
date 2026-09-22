
export function MethodSection() {
  const steps = [
    {
      n: "01",
      title: "Prepare",
      copy: "Audio is validated, normalized, and resampled to a consistent signal.",
      visual: (
        <div className="method-vis method-vis-prepare">
          <svg viewBox="0 0 100 40" className="vis-wave">
            <path d="M0 20 Q 25 5, 50 20 T 100 20" fill="none" stroke="currentColor" strokeWidth="2.5"/>
          </svg>
        </div>
      )
    },
    {
      n: "02",
      title: "Divide",
      copy: "Overlapping segments preserve where evidence appears in time.",
      visual: (
        <div className="method-vis method-vis-divide">
          <div className="divide-segments">
            <span /><span /><span /><span />
          </div>
        </div>
      )
    },
    {
      n: "03",
      title: "Represent",
      copy: "A speech model extracts representations from the audio itself.",
      visual: (
        <div className="method-vis method-vis-represent">
          <div className="nodes"><i/><i/><i/><i/></div>
        </div>
      )
    },
    {
      n: "04",
      title: "Compare",
      copy: "Each segment contributes evidence toward a human or synthetic classification.",
      visual: (
        <div className="method-vis method-vis-compare">
          <div className="bars-wrap">
            <div className="bar high" /><div className="bar low" /><div className="bar high" />
          </div>
        </div>
      )
    },
    {
      n: "05",
      title: "Interpret",
      copy: "Evidence is aggregated with uncertainty, signal quality, and caveats.",
      visual: (
        <div className="method-vis method-vis-interpret">
          <div className="gauge"><i className="needle" /></div>
        </div>
      )
    }
  ];

  return (
    <section className="method-section" id="method">
      {/* Visual Narrative Connector */}
      <div className="section-narrative-cue inverse method-cue" aria-hidden="true">
        <span className="narrative-line" />
        <span className="narrative-node">04 / THE ANALYSIS PIPELINE</span>
        <span className="narrative-line" />
      </div>

      <div className="section-intro inverse method-intro centered-intro">
        <h2>Not one guess.<br />A sequence of observations.</h2>
        <p className="section-intro-desc">
          Rather than reducing a recording to a single score, Vaak evaluates speech across time and surfaces the segments that contribute most strongly to its judgment.
        </p>
      </div>
      <div className="method-flow">
        {steps.map(({ n, title, copy, visual }) => (
          <article key={n}>
            <span className="step-num">{n}</span>
            {visual}
            <h3>{title}</h3>
            <p>{copy}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
