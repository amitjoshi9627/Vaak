export function ListeningMark({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`listening-mark ${compact ? "compact" : ""}`} aria-hidden="true">
      {[7, 14, 23, 35, 49, 62, 73, 82, 89, 94, 97, 98].map((size, index) => (
        <i key={size} style={{ width: `${size}%`, opacity: 0.18 + index * 0.065 }} />
      ))}
    </div>
  );
}
