import type { ResearchSelectionState } from "../../types/probeEvidence";

export type EpisodeOpenContext = {
  feature?: number | null;
  fromCohort?: boolean;
  inspectionMode?: string;
  lensRunId?: string;
  policyCall?: number | null;
  probeId?: string;
  rankingMode?: string;
  researchSelection?: ResearchSelectionState;
  siteName?: string;
};
