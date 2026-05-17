import { postJson } from "./client";
import type { AxisValues, LensArraySlice } from "../types/workbench";

export function sliceLensArray(
  arrayId: string,
  selection: AxisValues,
  maxValues = 4096,
): Promise<LensArraySlice> {
  return postJson<LensArraySlice>(`/api/lens-arrays/${encodeURIComponent(arrayId)}/slice`, {
    selection,
    max_values: maxValues,
  });
}
