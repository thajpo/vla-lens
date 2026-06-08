import type { WorkbenchManifest } from "../../types/workbench";
import type { InterventionLabSeed } from "../../types/interventions";
import type { InspectionMode } from "./shared";

export type EpisodeTraceChangeContext = {
  feature?: number | null;
  fromCohort?: boolean;
  inspectionMode?: InspectionMode | string;
  policyCall?: number | null;
  probeId?: string;
  rankingMode?: string;
  siteName?: string;
};

export type EpisodesPageProps = {
  cohortReturnHref?: string;
  initialFeature?: number;
  initialInspectionMode?: InspectionMode;
  initialLensRankingMode?: string;
  initialPolicyCall?: number;
  initialProbeArtifactId?: string;
  initialSiteName?: string;
  manifest?: WorkbenchManifest;
  initialTraceId?: string;
  onSendToIntervention?: (seed: InterventionLabSeed) => void;
  onTraceChange?: (traceId: string, context?: EpisodeTraceChangeContext) => void;
};
