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

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  return readJson<T>(response);
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
    const message = payload?.error?.message ?? response.statusText;
    throw new ApiError(message, response.status, payload);
  }
  return payload as T;
}
