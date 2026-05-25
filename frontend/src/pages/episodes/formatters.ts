import type {
  ExpertTokenDetailsResponse,
  PromptAttentionResponse,
  PromptTokenAttention,
  SelectedPatch,
} from "../../types/dataset";
import type { CameraOverlayPayload } from "./shared";

export function labelFromSnake(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function heatColor(ratio: number): string {
  const clamped = Math.max(0, Math.min(1, ratio));
  const hue = 205 - clamped * 175;
  const light = 92 - clamped * 34;
  return `hsl(${hue} 78% ${light}%)`;
}

export function overlayPatchValue(
  overlay: CameraOverlayPayload | undefined,
  patch: SelectedPatch,
): number | null {
  const value = overlay?.maps?.[patch.camera]?.values?.[patch.row]?.[patch.col];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function overlayCameraMaxAbs(
  overlay: CameraOverlayPayload | undefined,
  camera: string,
): number {
  const values = overlay?.maps?.[camera]?.values ?? [];
  const finite = values.flat().filter((value) => Number.isFinite(value));
  return Math.max(...finite.map((value) => Math.abs(value)), 1e-6);
}

export function orderedPromptAttentionRows(
  expertTokenDetails?: ExpertTokenDetailsResponse,
  promptAttention?: PromptAttentionResponse,
): PromptTokenAttention[] {
  if (expertTokenDetails?.available && expertTokenDetails.prompt_tokens?.length) {
    return expertTokenDetails.prompt_tokens;
  }
  if (promptAttention?.prompt_tokens?.length) {
    return promptAttention.prompt_tokens;
  }
  if (expertTokenDetails?.available && expertTokenDetails.top_prompt_tokens?.length) {
    return [...expertTokenDetails.top_prompt_tokens].sort(
      (left, right) => left.local_index - right.local_index,
    );
  }
  return [...(promptAttention?.top_text_tokens ?? [])].sort(
    (left, right) => left.local_index - right.local_index,
  );
}

export function taskPromptRows(rows: PromptTokenAttention[]): PromptTokenAttention[] {
  if (!rows.length) {
    return rows;
  }
  const labels = rows.map((row) => displayTokenPiece(row).trim());
  let start = 0;
  const taskIndex = labels.findIndex((label) => label === "Task" || label === "Task:");
  if (taskIndex >= 0) {
    const colonIndex = labels.findIndex((label, index) => index > taskIndex && label === ":");
    start = colonIndex >= 0 ? colonIndex + 1 : taskIndex + 1;
  } else if (labels[0] === "<bos>") {
    start = 1;
  }

  const stateIndex = labels.findIndex(
    (label, index) => index > start && (label === "State" || label === "State:"),
  );
  const actionIndex = labels.findIndex(
    (label, index) => index > start && (label === "Action" || label === "Action:"),
  );
  const stopCandidates = [stateIndex, actionIndex].filter((index) => index >= 0);
  let end = stopCandidates.length ? Math.min(...stopCandidates) : rows.length;
  while (end > start && labels[end - 1] === ",") {
    end -= 1;
  }
  return rows.slice(start, end).filter((row) => {
    const label = displayTokenPiece(row).trim();
    return label && label !== "<bos>" && label !== "<eos>";
  });
}

export function promptRowsMatchPrompt(rows: PromptTokenAttention[], prompt?: string | null): boolean {
  if (!rows.length || !prompt) {
    return true;
  }
  const reconstructed = normalizePromptText(rows.map(displayTokenPiece).join(""));
  const expected = normalizePromptText(prompt);
  return reconstructed === expected || reconstructed.includes(expected) || expected.includes(reconstructed);
}

export function normalizePromptText(value: string): string {
  return value.replace(/\s+/gu, " ").trim().toLowerCase();
}

export function displayTokenPiece(row: PromptTokenAttention): string {
  const token = promptTokenDisplay(row);
  const text = `${token.prefix}${token.text}`;
  return text || (row.token_id === null || row.token_id === undefined ? "?" : "token");
}

export function promptTokenTitle(row: PromptTokenAttention): string {
  const parts = [`token ${row.local_index}`, displayTokenPiece(row)];
  if (row.token_id !== null && row.token_id !== undefined) {
    parts.push(`id ${row.token_id}`);
  }
  return parts.join(" - ");
}

export function promptTokenDisplay(row: PromptTokenAttention): { prefix: string; text: string } {
  const rawPiece = row.token_piece;
  if (rawPiece === null || rawPiece === undefined || rawPiece === "") {
    return {
      prefix: "",
      text: row.token_id === null || row.token_id === undefined ? "?" : "",
    };
  }
  return cleanPromptTokenPiece(String(rawPiece));
}

export function cleanPromptTokenPiece(piece: string): { prefix: string; text: string } {
  let text = piece.replaceAll("<0x0A>", "\n").replaceAll("Ċ", "\n");
  let prefix = "";
  const boundary = text.match(/^[▁_]+/u);
  if (boundary) {
    prefix += " ";
    text = text.slice(boundary[0].length);
  }
  const leadingWhitespace = text.match(/^\s+/u);
  if (leadingWhitespace) {
    prefix += leadingWhitespace[0];
    text = text.slice(leadingWhitespace[0].length);
  }
  return { prefix, text };
}

export function signedActivationColor(value: number, maxAbs: number): string {
  const ratio = Math.min(1, Math.abs(value) / Math.max(maxAbs, 1e-6));
  if (value >= 0) {
    return `rgba(156, 74, 88, ${0.07 + ratio * 0.48})`;
  }
  return `rgba(47, 111, 127, ${0.07 + ratio * 0.44})`;
}

export function formatMaybeNumber(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "-";
}

export function formatVector(values: number[], precision: number): string {
  return `[${values
    .map((value) =>
      typeof value === "number" && Number.isFinite(value) ? value.toFixed(precision) : "-",
    )
    .join(", ")}]`;
}

export function formatPercent(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";
}

export function formatLayerNumber(layer: number): string {
  return Number.isInteger(layer) ? String(layer) : layer.toFixed(1);
}
