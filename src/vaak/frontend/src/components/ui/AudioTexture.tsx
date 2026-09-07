import { useMemo } from 'react';

export function AudioTexture({ active = false }: { active?: boolean }) {
  const bars = useMemo(() => Array.from({ length: 72 }, (_, i) => {
    const value = Math.abs(Math.sin(i * 1.47) * 28 + Math.cos(i * 0.39) * 18);
    return Math.max(8, Math.round(value));
  }), []);
  return (
    <div className={`audio-texture ${active ? "is-active" : ""}`} aria-hidden="true">
      {bars.map((height, i) => <i key={i} style={{ height: `${height}%`, animationDelay: `${(i % 11) * -0.08}s` }} />)}
    </div>
  );
}
