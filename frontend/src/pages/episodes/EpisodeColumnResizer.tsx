import type { PointerEvent as ReactPointerEvent } from "react";

type EpisodeColumnResizerProps = {
  onResizePctChange: (value: number) => void;
};

export function EpisodeColumnResizer({ onResizePctChange }: EpisodeColumnResizerProps) {
  const handleResize = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const root = event.currentTarget.closest(".episodes-workspace");
    const rect = root?.getBoundingClientRect();
    if (!rect) {
      return;
    }
    const rightWidth = rect.right - event.clientX;
    const nextPct = (rightWidth / rect.width) * 100;
    onResizePctChange(Math.max(30, Math.min(62, nextPct)));
  };
  return (
    <button
      aria-label="Resize model inspector"
      className="episode-column-resizer"
      type="button"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId);
        handleResize(event);
      }}
      onPointerMove={(event) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          handleResize(event);
        }
      }}
      onPointerUp={(event) => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) {
          event.currentTarget.releasePointerCapture(event.pointerId);
        }
      }}
      onDoubleClick={() => onResizePctChange(38)}
    />
  );
}
