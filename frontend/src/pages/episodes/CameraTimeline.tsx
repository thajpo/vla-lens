import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchObjectCameraOverlay,
  frameUrl,
} from "../../api/dataset";
import type {
  ObjectCameraOverlayObject,
  SelectedPatch,
} from "../../types/dataset";
import type { CameraOverlayPayload, CameraOverlayStatus } from "./shared";
import {
  formatVector,
  signedActivationColor,
} from "./formatters";

export { FramePlaybackControls } from "./FramePlaybackControls";

export function CameraGrid({
  cacheKey,
  cameras,
  imageTokenMap,
  isPlaying,
  maxTimestep,
  overlayStatus,
  preloadFrameCount = 2,
  showAttentionOverlay,
  showObjectOverlay,
  onPatchSelect,
  selectedPatch,
  traceId,
  timestep,
}: {
  cacheKey: string;
  cameras: string[];
  imageTokenMap?: CameraOverlayPayload;
  isPlaying: boolean;
  maxTimestep: number;
  overlayStatus?: CameraOverlayStatus;
  preloadFrameCount?: number;
  showAttentionOverlay: boolean;
  showObjectOverlay: boolean;
  onPatchSelect: (patch: SelectedPatch | null) => void;
  selectedPatch: SelectedPatch | null;
  traceId: string;
  timestep: number;
}) {
  const preloadedFrames = useRef<Set<string>>(new Set());
  const preloadImages = useRef<HTMLImageElement[]>([]);

  useEffect(() => {
    preloadedFrames.current.clear();
    preloadImages.current = [];
  }, [cacheKey, traceId]);

  useEffect(() => {
    if (!cameras.length || maxTimestep <= timestep) {
      return;
    }
    if (!isPlaying) {
      return;
    }
    const lookahead = Math.max(1, Math.min(10, preloadFrameCount));
    for (let next = timestep + 1; next <= Math.min(maxTimestep, timestep + lookahead); next += 1) {
      for (const camera of cameras) {
        const key = `${cacheKey}:${traceId}:${camera}:${next}`;
        if (preloadedFrames.current.has(key)) {
          continue;
        }
        preloadedFrames.current.add(key);
        const image = new Image();
        image.decoding = "async";
        image.src = frameUrl(traceId, camera, next, cacheKey);
        void image.decode?.().catch(() => undefined);
        preloadImages.current.push(image);
      }
    }
    if (preloadImages.current.length > 160) {
      preloadImages.current = preloadImages.current.slice(-160);
    }
  }, [cacheKey, cameras, isPlaying, maxTimestep, preloadFrameCount, timestep, traceId]);

  if (!cameras.length) {
    return <div className="empty-state">No camera streams for this episode.</div>;
  }
  return (
    <div className="camera-grid">
      {cameras.map((camera) => (
        <CameraFrame
          cacheKey={cacheKey}
          camera={camera}
          imageTokenMap={imageTokenMap}
          isPlaying={isPlaying}
          overlayStatus={overlayStatus}
          showAttentionOverlay={showAttentionOverlay}
          showObjectOverlay={showObjectOverlay}
          key={camera}
          onPatchSelect={onPatchSelect}
          selectedPatch={selectedPatch}
          timestep={timestep}
          traceId={traceId}
        />
      ))}
    </div>
  );
}

function CameraFrame({
  cacheKey,
  camera,
  imageTokenMap,
  isPlaying,
  overlayStatus,
  showAttentionOverlay,
  showObjectOverlay,
  onPatchSelect,
  selectedPatch,
  timestep,
  traceId,
}: {
  cacheKey: string;
  camera: string;
  imageTokenMap?: CameraOverlayPayload;
  isPlaying: boolean;
  overlayStatus?: CameraOverlayStatus;
  showAttentionOverlay: boolean;
  showObjectOverlay: boolean;
  onPatchSelect: (patch: SelectedPatch | null) => void;
  selectedPatch: SelectedPatch | null;
  timestep: number;
  traceId: string;
}) {
  const frameSrc = frameUrl(traceId, camera, timestep, cacheKey);
  const [loadedFrameSrc, setLoadedFrameSrc] = useState("");
  const frameReady = loadedFrameSrc === frameSrc;
  const objectOverlay = useQuery({
    queryKey: ["object-camera-overlay", traceId, camera, timestep],
    queryFn: ({ signal }) => fetchObjectCameraOverlay(traceId, camera, timestep, signal),
    enabled: showObjectOverlay && !isPlaying,
    staleTime: 60_000,
  });
  const visibleObjects = useMemo(
    () => {
      if (
        !showObjectOverlay ||
        isPlaying ||
        !frameReady ||
        objectOverlay.data?.timestep !== timestep
      ) {
        return [];
      }
      return (objectOverlay.data?.objects ?? []).filter(
        (object) =>
          object.in_frame &&
          Number.isFinite(object.x ?? NaN) &&
          Number.isFinite(object.y ?? NaN),
      );
    },
    [frameReady, isPlaying, objectOverlay.data?.objects, objectOverlay.data?.timestep, showObjectOverlay, timestep],
  );
  const [hoveredObject, setHoveredObject] = useState<ObjectCameraOverlayObject | null>(null);
  const [imageHovered, setImageHovered] = useState(false);

  const activeHoveredObject = isPlaying || !frameReady || !showObjectOverlay ? null : hoveredObject;
  const showOverlayStatus = Boolean(
    showAttentionOverlay &&
      overlayStatus &&
      (imageHovered || overlayStatus.isUpdating || overlayStatus.isStale || imageTokenMap?.available === false),
  );

  return (
    <figure className="camera-frame">
      <div
        className="camera-image-wrap"
        onPointerEnter={() => setImageHovered(true)}
        onPointerMove={(event) => {
          if (isPlaying) {
            setHoveredObject((current) => current === null ? current : null);
            return;
          }
          if (!visibleObjects.length) {
            setHoveredObject((current) => current === null ? current : null);
            return;
          }
          const rect = event.currentTarget.getBoundingClientRect();
          const x = event.clientX - rect.left;
          const y = event.clientY - rect.top;
          const normX = x / Math.max(1, rect.width);
          const normY = y / Math.max(1, rect.height);
          let containing: ObjectCameraOverlayObject | null = null;
          let containingArea = Number.POSITIVE_INFINITY;
          for (const object of visibleObjects) {
            const bbox = normalizedObjectBbox(object);
            if (!bbox) {
              continue;
            }
            if (normX < bbox.x0 || normX > bbox.x1 || normY < bbox.y0 || normY > bbox.y1) {
              continue;
            }
            const area = Math.max(0, bbox.x1 - bbox.x0) * Math.max(0, bbox.y1 - bbox.y0);
            if (area < containingArea) {
              containing = object;
              containingArea = area;
            }
          }
          if (containing) {
            setHoveredObject((current) => sameOverlayObject(current, containing) ? current : containing);
            return;
          }
          let nearest: ObjectCameraOverlayObject | null = null;
          let nearestDistance = Number.POSITIVE_INFINITY;
          for (const object of visibleObjects) {
            const objectX = Number(object.x) * rect.width;
            const objectY = Number(object.y) * rect.height;
            const distance = Math.hypot(objectX - x, objectY - y);
            if (distance < nearestDistance) {
              nearest = object;
              nearestDistance = distance;
            }
          }
          const nextHoveredObject = nearestDistance <= 44 ? nearest : null;
          setHoveredObject((current) => sameOverlayObject(current, nextHoveredObject) ? current : nextHoveredObject);
        }}
        onPointerLeave={() => {
          setImageHovered(false);
          setHoveredObject((current) => current === null ? current : null);
        }}
      >
        <img
          alt={`${traceId} ${camera} timestep ${timestep}`}
          decoding="async"
          src={frameSrc}
          onLoad={() => setLoadedFrameSrc(frameSrc)}
        />
        <ObjectCameraOverlay
          objects={visibleObjects}
          hoveredObject={activeHoveredObject}
          showMarks={showObjectOverlay}
        />
        {showAttentionOverlay && (isPlaying || frameReady) ? (
          <ActivationGridOverlay
            camera={camera}
            imageTokenMap={imageTokenMap}
            selectedPatch={selectedPatch}
            stale={Boolean(overlayStatus?.isStale)}
            onPatchSelect={onPatchSelect}
          />
        ) : null}
        {showOverlayStatus && overlayStatus ? (
          <OverlayStatusBadge
            available={Boolean(imageTokenMap?.available)}
            status={overlayStatus}
          />
        ) : null}
      </div>
      <figcaption>
        <span>{camera}</span>
        <span>t={timestep}</span>
      </figcaption>
    </figure>
  );
}

function ObjectCameraOverlay({
  hoveredObject,
  objects,
  showMarks,
}: {
  hoveredObject: ObjectCameraOverlayObject | null;
  objects: ObjectCameraOverlayObject[];
  showMarks: boolean;
}) {
  if (!objects.length) {
    return null;
  }
  return (
    <div className="object-camera-overlay" aria-hidden="true">
      {showMarks ? objects.map((object) => {
        const bbox = normalizedObjectBbox(object);
        const active = hoveredObject?.object_index === object.object_index;
        if (!bbox) {
          return null;
        }
        return (
          <span
            className={`object-camera-bbox${active ? " active" : ""}`}
            key={`bbox:${object.object_index}:${object.object_name}`}
            style={{
              left: `${bbox.x0 * 100}%`,
              top: `${bbox.y0 * 100}%`,
              width: `${Math.max(0, bbox.x1 - bbox.x0) * 100}%`,
              height: `${Math.max(0, bbox.y1 - bbox.y0) * 100}%`,
            }}
          />
        );
      }) : null}
      {objects.map((object) => {
        const x = Number(object.x);
        const y = Number(object.y);
        const left = `${Math.min(98, Math.max(2, x * 100))}%`;
        const top = `${Math.min(98, Math.max(2, y * 100))}%`;
        const active = hoveredObject?.object_index === object.object_index;
        if (!showMarks && !active) {
          return null;
        }
        const edgeClass = objectOverlayEdgeClass(x, y);
        return (
          <div
            className={["object-camera-marker", active ? "active" : "", edgeClass]
              .filter(Boolean)
              .join(" ")}
            key={`${object.object_index}:${object.object_name}`}
            style={{ left, top }}
          >
            {showMarks ? <span className="object-camera-dot" /> : null}
            {active ? <span className="object-camera-label">{object.object_name}</span> : null}
          </div>
        );
      })}
      {hoveredObject ? (
        <div
          className={["object-camera-tooltip", objectOverlayEdgeClass(
            Number(hoveredObject.x),
            Number(hoveredObject.y),
          )]
            .filter(Boolean)
            .join(" ")}
          style={{
            left: `${Math.min(96, Math.max(4, Number(hoveredObject.x) * 100))}%`,
            top: `${Math.min(94, Math.max(4, Number(hoveredObject.y) * 100))}%`,
          }}
        >
          <strong>{hoveredObject.object_name}</strong>
          <span>{hoveredObject.object_kind ?? "object"}</span>
          {hoveredObject.position_world ? (
            <span>xyz {formatVector(hoveredObject.position_world, 3)}</span>
          ) : null}
          {hoveredObject.geometry_center_world ? (
            <span>geom {formatVector(hoveredObject.geometry_center_world, 3)}</span>
          ) : null}
          {hoveredObject.quaternion_xyzw ? (
            <span>quat {formatVector(hoveredObject.quaternion_xyzw, 3)}</span>
          ) : null}
          {hoveredObject.projection_kind ? <span>{hoveredObject.projection_kind}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

function objectOverlayEdgeClass(x: number, y: number) {
  return [
    x > 0.72 ? "edge-right" : "",
    x < 0.18 ? "edge-left" : "",
    y > 0.72 ? "edge-bottom" : "",
  ]
    .filter(Boolean)
    .join(" ");
}

function sameOverlayObject(
  left: ObjectCameraOverlayObject | null,
  right: ObjectCameraOverlayObject | null,
): boolean {
  return left === right;
}

function normalizedObjectBbox(object: ObjectCameraOverlayObject) {
  const bbox = object.bbox;
  if (!bbox) {
    return null;
  }
  const x0 = Number(bbox.x0);
  const y0 = Number(bbox.y0);
  const x1 = Number(bbox.x1);
  const y1 = Number(bbox.y1);
  if (![x0, y0, x1, y1].every(Number.isFinite)) {
    return null;
  }
  return {
    x0: Math.max(0, Math.min(1, Math.min(x0, x1))),
    y0: Math.max(0, Math.min(1, Math.min(y0, y1))),
    x1: Math.max(0, Math.min(1, Math.max(x0, x1))),
    y1: Math.max(0, Math.min(1, Math.max(y0, y1))),
  };
}

function ActivationGridOverlay({
  camera,
  imageTokenMap,
  onPatchSelect,
  selectedPatch,
  stale,
}: {
  camera: string;
  imageTokenMap?: CameraOverlayPayload;
  onPatchSelect: (patch: SelectedPatch | null) => void;
  selectedPatch: SelectedPatch | null;
  stale: boolean;
}) {
  const cameraMapValues = imageTokenMap?.maps?.[camera]?.values;
  const values = useMemo(() => cameraMapValues ?? [], [cameraMapValues]);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [layoutVersion, setLayoutVersion] = useState(0);

  useEffect(() => {
    const target = canvasRef.current?.parentElement;
    if (!target) {
      return;
    }
    const observer = new ResizeObserver(() => {
      setLayoutVersion((version) => version + 1);
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageTokenMap?.available || !values.length) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    const scale = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(rect.width * scale));
    canvas.height = Math.max(1, Math.round(rect.height * scale));

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const flat = values.flat().filter((value) => Number.isFinite(value));
    const maxAbs = Math.max(...flat.map((value) => Math.abs(value)), 1e-6);
    const rows = values.length;
    const cols = Math.max(1, values[0]?.length ?? 1);
    const cellWidth = rect.width / cols;
    const cellHeight = rect.height / rows;

    values.forEach((row, rowIndex) => {
      row.forEach((value, colIndex) => {
        if (!Number.isFinite(value)) {
          return;
        }
        ctx.fillStyle = signedActivationColor(value, maxAbs);
        ctx.fillRect(colIndex * cellWidth, rowIndex * cellHeight, cellWidth, cellHeight);
      });
    });

    ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
    ctx.lineWidth = 1;
    for (let row = 1; row < rows; row += 1) {
      const y = row * cellHeight;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(rect.width, y);
      ctx.stroke();
    }
    for (let col = 1; col < cols; col += 1) {
      const x = col * cellWidth;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, rect.height);
      ctx.stroke();
    }

    if (selectedPatch?.camera === camera) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.98)";
      ctx.lineWidth = 3;
      ctx.strokeRect(
        selectedPatch.col * cellWidth + 1.5,
        selectedPatch.row * cellHeight + 1.5,
        Math.max(0, cellWidth - 3),
        Math.max(0, cellHeight - 3),
      );
      ctx.strokeStyle = "rgba(15, 23, 42, 0.88)";
      ctx.lineWidth = 1;
      ctx.strokeRect(
        selectedPatch.col * cellWidth + 4,
        selectedPatch.row * cellHeight + 4,
        Math.max(0, cellWidth - 8),
        Math.max(0, cellHeight - 8),
      );
    }
  }, [camera, imageTokenMap?.available, layoutVersion, selectedPatch, values]);

  if (stale || !imageTokenMap?.available || !values.length) {
    return null;
  }
  return (
    <canvas
      ref={canvasRef}
      className="activation-grid-overlay"
      aria-label={`${camera} activation heatmap overlay`}
      role="button"
      tabIndex={0}
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const cols = Math.max(1, values[0]?.length ?? 1);
        const rows = values.length;
        const col = Math.max(
          0,
          Math.min(cols - 1, Math.floor(((event.clientX - rect.left) / rect.width) * cols)),
        );
        const row = Math.max(
          0,
          Math.min(rows - 1, Math.floor(((event.clientY - rect.top) / rect.height) * rows)),
        );
        onPatchSelect({ camera, row, col });
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onPatchSelect({ camera, row: 0, col: 0 });
        }
      }}
    />
  );
}

function OverlayStatusBadge({
  available,
  status,
}: {
  available: boolean;
  status: CameraOverlayStatus;
}) {
  return (
    <div
      className={[
        "camera-overlay-status",
        status.isUpdating ? "updating" : "",
        status.isStale ? "stale" : "",
        !available ? "unavailable" : "",
      ].filter(Boolean).join(" ")}
    >
      <strong>{status.isUpdating ? "Updating" : status.label}</strong>
      {status.detail ? <span>{status.detail}</span> : null}
    </div>
  );
}
