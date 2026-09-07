export type AppState = "empty" | "ready" | "recording" | "processing" | "result" | "error";
export type Evidence = { start: number; end: number; strength: "high" | "moderate" };

export const formatTime = (seconds: number) => {
  const safe = Number.isFinite(seconds) ? seconds : 0;
  const minutes = Math.floor(safe / 60);
  return `${minutes}:${(safe % 60).toFixed(1).padStart(4, "0")}`;
};
