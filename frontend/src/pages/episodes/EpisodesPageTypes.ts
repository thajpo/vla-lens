import type { WorkbenchManifest } from "../../types/workbench";
import type { InterventionLabSeed } from "../../types/interventions";
import type { ResearchSelectionState } from "../../types/probeEvidence";
import type { InspectionMode } from "./shared";

export type EpisodeTraceChangeContext = {
  feature?: number | null;
  fromCohort?: boolean;
  inspectionMode?: InspectionMode | string;
  lensRunId?: string;
  policyCall?: number | null;
  probeId?: string;
  rankingMode?: string;
  researchSelection?: ResearchSelectionState;
  siteName?: string;
};

export type EpisodesPageProps = {
  cohortReturnHref?: string;
  initialFeature?: number;
  initialInspectionMode?: InspectionMode;
  initialLensRunId?: string;
  initialLensRankingMode?: string;
  initialPolicyCall?: number;
  initialProbeArtifactId?: string;
  initialResearchSelection?: ResearchSelectionState;
  initialSiteName?: string;
  initialTimestep?: number;
  manifest?: WorkbenchManifest;
  initialTraceId?: string;
  onSendToIntervention?: (seed: InterventionLabSeed) => void;
  onTraceChange?: (traceId: string, context?: EpisodeTraceChangeContext) => void;
};
