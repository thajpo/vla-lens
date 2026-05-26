export class ApiError extends Error {
  status: number;
  payload: unknown;

  constructor(message: string, status: number, payload: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

export type JsonRequestInit = RequestInit & {
  cache?: RequestCache;
};

export async function getJson<T>(path: string, init?: JsonRequestInit): Promise<T> {
  const response = await fetch(path, init);
  return readJson<T>(response);
}

export function noStore(init: JsonRequestInit = {}): JsonRequestInit {
  return { ...init, cache: "no-store" };
}

export async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return readJson<T>(response);
}

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok || payload?.error) {
    const message = apiErrorMessage(payload, response.statusText);
    throw new ApiError(message, response.status, payload);
  }
  return payload as T;
}

function apiErrorMessage(payload: unknown, fallback: string): string {
  if (isRecord(payload)) {
    if (typeof payload.error === "string" && payload.error) {
      return typeof payload.message === "string" && payload.message
        ? payload.message
        : payload.error;
    }
    if (isRecord(payload.error) && typeof payload.error.message === "string" && payload.error.message) {
      return payload.error.message;
    }
    if (typeof payload.message === "string" && payload.message) {
      return payload.message;
    }
  }
  return fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
