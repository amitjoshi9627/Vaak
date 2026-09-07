/**
 * Topbar — strict, bordered, utilitarian.
 * Sits at the top of the entire application.
 */
export function Topbar() {
  return (
    <nav
      id="topbar"
      style={{
        height: 'var(--topbar-h)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 var(--space-6)',
        borderBottom: '1px solid var(--grid-line)',
        background: 'var(--bg-paper)',
        position: 'sticky',
        top: 0,
        zIndex: 100,
      }}
    >
      {/* Brand Group */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span
            style={{
              fontFamily: 'var(--font-serif)',
              fontWeight: 600,
              fontSize: 22,
              letterSpacing: '0.04em',
              color: 'var(--ink-primary)',
            }}
          >
            VAAK
          </span>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              color: 'var(--ink-muted)',
              letterSpacing: '0.05em',
            }}
          >
            [वाक्]
          </span>
        </div>
        <div style={{ width: 1, height: 16, background: 'var(--grid-line)' }} />
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            color: 'var(--ink-secondary)',
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
          }}
        >
          Acoustic Forensic Engine
        </span>
      </div>

      {/* System status pill */}
      <div 
        style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 12,
          padding: '4px 12px',
          border: '1px solid var(--grid-line-strong)',
        }}
      >
        <span
          style={{
            display: 'inline-block',
            width: 6,
            height: 6,
            borderRadius: '50%',
            background: 'var(--ink-primary)',
            animation: 'blink 2s step-end infinite',
          }}
        />
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 9,
            letterSpacing: '0.12em',
            textTransform: 'uppercase',
            color: 'var(--ink-primary)',
            fontWeight: 500,
          }}
        >
          ENGINE: WAVLM-BASE ACTIVE
        </span>
      </div>
    </nav>
  );
}
