export function IntroAcousticGraphic() {
  return (
    <div className="acoustic-signal-overlay" aria-hidden="true">
      <svg 
        className="acoustic-signal-svg" 
        viewBox="0 0 800 600" 
        fill="none" 
        preserveAspectRatio="none"
      >
        {/* Layer 3: Upper harmonic articulation contour (Warm Paper) */}
        <g className="trace-group trace-group-3">
          <path
            d="M -20 280 C 100 280, 160 325, 250 315 C 340 305, 410 365, 510 360 C 610 355, 700 250, 820 265"
            className="signal-trace trace-3"
            stroke="#f5f0e6"
            strokeWidth="1.6"
            strokeLinecap="round"
            opacity="0.72"
          />
        </g>

        {/* Layer 2: Secondary formant resonance contour (Soft Sage / Mint accent) */}
        <g className="trace-group trace-group-2">
          <path
            d="M -20 315 C 110 315, 180 265, 270 270 C 360 275, 440 220, 540 235 C 640 250, 710 320, 820 310"
            className="signal-trace trace-2"
            stroke="#c8dad0"
            strokeWidth="2.0"
            strokeLinecap="round"
            opacity="0.86"
          />
        </g>

        {/* Layer 1: Primary fundamental speech contour (Luminous Warm Ivory) */}
        <g className="trace-group trace-group-1">
          <path
            d="M -20 295 C 130 295, 190 235, 290 240 C 390 245, 430 335, 530 325 C 630 315, 690 265, 820 275"
            className="signal-trace trace-1"
            stroke="#fdfbf7"
            strokeWidth="2.6"
            strokeLinecap="round"
            opacity="0.95"
          />
        </g>
      </svg>
    </div>
  );
}
