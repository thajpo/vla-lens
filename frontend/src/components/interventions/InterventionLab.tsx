import { AlertTriangle, FlaskConical, Play, Save } from "lucide-react";
import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchArtifacts, fetchDataset, fetchEpisodesPage } from "../../api/dataset";
import {
  preflightIntervention,
  saveInterventionRun,
} from "../../api/interventions";
import type { ArtifactRecord, DatasetEpisode } from "../../types/dataset";
import type {
  InterventionLabDraft,
  InterventionLabSeed,
  InterventionPreflightResult,
} from "../../types/interventions";
import {
  buildInspectedInterventionRecord,
  buildInterventionRequest,
  liveRunAvailable,
} from "./interventionLabModel";
import { statusClass } from "./interventionDisplay";

type InterventionLabProps = {
  initialDraft?: InterventionLabSeed;
  onSavedRun: (runId: string) => void;
};

type LabModelSite = {
  family?: string | null;
  layer?: number | null;
  site_id: string;
  token_kind?: string | null;
  token_space_id?: string | null;
};

export function InterventionLab({ initialDraft, onSavedRun }: InterventionLabProps) {
  const queryClient = useQueryClient();
  const dataset = useQuery({ queryKey: ["dataset"], queryFn: fetchDataset, staleTime: 15_000 });
  const artifacts = useQuery({ queryKey: ["artifacts"], queryFn: fetchArtifacts, staleTime: 15_000 });
  const episodes = useQuery({
    queryKey: ["episodes", "intervention-lab"],
    queryFn: () => fetchEpisodesPage({ limit: 100 }),
    staleTime: 15_000,
  });
  const probeArtifacts = useMemo(
    () => (artifacts.data?.artifacts ?? []).filter((artifact) => artifact.artifact_type === "probe_suite"),
    [artifacts.data],
  );
  const [artifactId, setArtifactId] = useState(initialDraft?.artifactId ?? "");
  const [traceId, setTraceId] = useState(initialDraft?.traceId ?? "");
  const [policyCallIndex, setPolicyCallIndex] = useState(initialDraft?.policyCallIndex ?? 0);
  const [modelSite, setModelSite] = useState(initialDraft?.modelSite ?? "");
  const [operator, setOperator] = useState(initialDraft?.operator ?? "add_direction");
  const [strength, setStrength] = useState(initialDraft?.strength ?? 1);
  const [basis, setBasis] = useState(initialDraft?.basis ?? ["raw", "gripper"]);
  const [includeRandomControl, setIncludeRandomControl] = useState(true);
  const [message, setMessage] = useState("");
  const [seedTarget, setSeedTarget] = useState<Record<string, unknown> | undefined>(initialDraft?.target);
  const modelSites = useMemo(
    () => interventionModelSites(initialDraft, seedTarget),
    [initialDraft, seedTarget],
  );

  const activeArtifactId = artifactId;
  const selectedArtifact = probeArtifacts.find(
    (artifact) => artifact.artifact_id === activeArtifactId,
  );
  const activeModelSite = modelSite || modelSites[0]?.site_id || "";
  const selectedModelSite = modelSites.find((site) => site.site_id === activeModelSite);
  const activeTraceId =
    traceId || firstSourceTrace(selectedArtifact) || episodes.data?.episodes[0]?.trace_id || "";
  const draft = buildDraft({
    artifact: selectedArtifact,
    artifactId: activeArtifactId,
    artifactType: initialDraft?.artifactType,
    basis,
    controls: includeRandomControl ? ["random_direction"] : [],
    datasetFingerprint: dataset.data?.index?.dataset_fingerprint ?? "unknown",
    datasetId: dataset.data?.root ?? "dataset",
    feature: initialDraft?.feature,
    layer: initialDraft?.layer,
    modelSite: activeModelSite,
    operator,
    policyCallIndex,
    rankingMode: initialDraft?.rankingMode,
    selectionSource: initialDraft?.selectionSource,
    site: selectedModelSite,
    sourceObjectRef: initialDraft?.sourceObjectRef,
    strength,
    target: seedTarget,
    title: initialDraft?.title,
    timestep: initialDraft?.timestep,
    traceId: activeTraceId,
  });
  const canPreflight = Boolean(draft.traceId && draft.modelSite);
  const preflight = useMutation({
    mutationFn: (payload: Record<string, unknown>) => preflightIntervention(payload),
    onSuccess: () => setMessage(""),
    onError: (error) => setMessage(error instanceof Error ? error.message : "Preflight failed"),
  });
  const save = useMutation({
    mutationFn: (record: Record<string, unknown>) => saveInterventionRun(record),
    onSuccess: async (response) => {
      await queryClient.invalidateQueries({ queryKey: ["intervention-runs"] });
      onSavedRun(response.intervention_run.run_id);
      setMessage("Saved");
    },
    onError: (error) => setMessage(error instanceof Error ? error.message : "Save failed"),
  });
  const preflightResult = preflight.data?.preflight;
  const runAvailable = liveRunAvailable(preflightResult);

  return (
    <section className="intervention-lab">
      <header className="intervention-lab-header">
        <div>
          <span>Intervention Lab</span>
          <h2>Probe Direction</h2>
        </div>
        <div className="intervention-lab-actions">
          <button
            className="secondary-button"
            disabled={!canPreflight || preflight.isPending}
            onClick={() => preflight.mutate(buildInterventionRequest(draft))}
            type="button"
          >
            <FlaskConical size={15} />
            Preflight
          </button>
          <button
            className="secondary-button"
            disabled={!runAvailable}
            onClick={() => setMessage("Live run route unavailable")}
            type="button"
          >
            <Play size={15} />
            Run
          </button>
          <button
            className="primary-button"
            disabled={!preflightResult || save.isPending}
            onClick={() => save.mutate(buildInspectedInterventionRecord(
              draft,
              preflightResult as InterventionPreflightResult,
              new Date().toISOString(),
            ))}
            type="button"
          >
            <Save size={15} />
            Save
          </button>
        </div>
      </header>

      <div className="intervention-lab-grid">
        <LabeledSelect
          label="Signal"
          onChange={(value) => {
            setArtifactId(value);
            setSeedTarget(undefined);
          }}
          options={[
            { label: "Manual model target", value: "" },
            ...probeArtifacts.map((artifact) => ({
              label: artifact.name ?? artifact.artifact_id ?? "probe",
              value: String(artifact.artifact_id ?? ""),
            })),
          ]}
          value={activeArtifactId}
        />
        <LabeledSelect
          label="Episode"
          onChange={(value) => {
            setTraceId(value);
            setSeedTarget(undefined);
          }}
          options={episodeOptions(episodes.data?.episodes ?? [], selectedArtifact)}
          value={activeTraceId}
        />
        <label className="lab-field">
          <span>Policy Call</span>
          <input
            min={0}
            onChange={(event) => {
              setPolicyCallIndex(Number(event.target.value) || 0);
              setSeedTarget(undefined);
            }}
            type="number"
            value={policyCallIndex}
          />
        </label>
        <LabeledSelect
          label="Site"
          onChange={(value) => {
            setModelSite(value);
            setSeedTarget(undefined);
          }}
          options={modelSites.map((site) => ({
            label: site.site_id,
            value: site.site_id,
          }))}
          value={activeModelSite}
        />
        <LabeledSelect
          label="Operator"
          onChange={setOperator}
          options={[
            { label: "Add direction", value: "add_direction" },
            { label: "Project out", value: "project_out_direction" },
            { label: "Ablate", value: "ablate" },
          ]}
          value={operator}
        />
        <label className="lab-field">
          <span>Strength</span>
          <input
            onChange={(event) => setStrength(Number(event.target.value) || 0)}
            step="0.25"
            type="number"
            value={strength}
          />
        </label>
      </div>

      <div className="intervention-lab-toggles">
        <label>
          <input
            checked={basis.includes("raw")}
            onChange={() => setBasis(toggleValue(basis, "raw"))}
            type="checkbox"
          />
          Raw action
        </label>
        <label>
          <input
            checked={basis.includes("gripper")}
            onChange={() => setBasis(toggleValue(basis, "gripper"))}
            type="checkbox"
          />
          Gripper
        </label>
        <label>
          <input
            checked={includeRandomControl}
            onChange={() => setIncludeRandomControl((value) => !value)}
            type="checkbox"
          />
          Random direction control
        </label>
      </div>

      {preflightResult ? <PreflightPanel preflight={preflightResult} /> : null}
      {message ? <p className="intervention-lab-message">{message}</p> : null}
    </section>
  );
}

function PreflightPanel({ preflight }: { preflight: InterventionPreflightResult }) {
  return (
    <section className="intervention-preflight-panel">
      <header>
        <span className={statusClass(preflight.status)}>{preflight.status}</span>
        <strong>{preflight.missing_capabilities.length} missing</strong>
      </header>
      <div className="preflight-checks">
        {preflight.checks.map((check) => (
          <div className="preflight-check" key={check.name}>
            <span>{check.name}</span>
            <strong>{check.status}</strong>
            {check.message ? <p>{check.message}</p> : null}
          </div>
        ))}
      </div>
      {preflight.errors.length ? (
        <div className="preflight-errors">
          <AlertTriangle size={15} />
          <span>{preflight.errors.join(" ")}</span>
        </div>
      ) : null}
    </section>
  );
}

function LabeledSelect({
  label,
  onChange,
  options,
  value,
}: {
  label: string;
  onChange: (value: string) => void;
  options: { label: string; value: string }[];
  value: string;
}) {
  const resolvedOptions = options.some((option) => option.value === value) || !value
    ? options
    : [{ label: value, value }, ...options];
  return (
    <label className="lab-field">
      <span>{label}</span>
      <select onChange={(event) => onChange(event.target.value)} value={value}>
        {!resolvedOptions.length ? <option value="">None</option> : null}
        {resolvedOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function buildDraft({
  artifact,
  artifactId,
  artifactType,
  basis,
  controls,
  datasetFingerprint,
  datasetId,
  feature,
  layer,
  modelSite,
  operator,
  policyCallIndex,
  rankingMode,
  selectionSource,
  site,
  sourceObjectRef,
  strength,
  target,
  title,
  timestep,
  traceId,
}: {
  artifact?: ArtifactRecord;
  artifactId: string;
  artifactType?: string;
  basis: string[];
  controls: string[];
  datasetFingerprint: string;
  datasetId: string;
  feature?: number | null;
  layer?: number | null;
  modelSite: string;
  operator: string;
  policyCallIndex: number;
  rankingMode?: string;
  selectionSource?: string;
  site?: LabModelSite;
  sourceObjectRef?: InterventionLabDraft["sourceObjectRef"];
  strength: number;
  target?: Record<string, unknown>;
  title?: string;
  timestep?: number | null;
  traceId: string;
}): InterventionLabDraft {
  return {
    artifactId: artifactId || String(artifact?.artifact_id ?? ""),
    artifactType: artifactType ?? String(artifact?.artifact_type ?? ""),
    basis,
    controls,
    datasetFingerprint,
    datasetId,
    feature,
    layer: layer ?? site?.layer ?? null,
    modelFamily: String(site?.family ?? "pi05"),
    modelSite,
    operator,
    policyCallIndex,
    rankingMode,
    selectionSource,
    sourceObjectRef,
    strength,
    target,
    title: title ?? (artifact?.name ? `Intervene with ${artifact.name}` : undefined),
    tokenSpace: String(site?.token_space_id ?? "pi05.action_suffix"),
    traceId,
    timestep,
  };
}

function interventionModelSites(
  draft?: InterventionLabSeed,
  target?: Record<string, unknown>,
): LabModelSite[] {
  const sites: LabModelSite[] = [];
  const targetSite = textValue(target?.model_site) || textValue(target?.site_id);
  const draftSite = draft?.modelSite ?? "";
  addSite(sites, targetSite || draftSite, {
    family: textValue(target?.model_family) || draft?.modelFamily,
    layer: numericValue(target?.layer) ?? draft?.layer ?? null,
    token_space_id: textValue(target?.token_space) || draft?.tokenSpace || "pi05.action_suffix",
  });
  addSite(sites, draft?.sourceObjectRef?.modelSite ?? "", {
    family: draft?.modelFamily,
    layer: draft?.sourceObjectRef?.layer ?? draft?.layer ?? null,
    token_space_id: draft?.tokenSpace || "pi05.action_suffix",
  });
  return sites.sort((left, right) => siteRank(left) - siteRank(right));
}

function addSite(sites: LabModelSite[], siteId: string, site: Omit<LabModelSite, "site_id">) {
  if (!siteId || sites.some((item) => item.site_id === siteId)) {
    return;
  }
  sites.push({ ...site, site_id: siteId });
}

function siteRank(site: LabModelSite): number {
  const id = site.site_id.toLowerCase();
  if (id.includes("action_head") || site.token_kind === "action") {
    return 0;
  }
  if (id.includes("expert")) {
    return 1;
  }
  return 2;
}

function textValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function numericValue(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function episodeOptions(episodes: DatasetEpisode[], artifact?: ArtifactRecord) {
  const seen = new Set<string>();
  const options: { label: string; value: string }[] = [];
  for (const traceId of artifact?.source_trace_ids ?? []) {
    if (traceId && !seen.has(traceId)) {
      seen.add(traceId);
      options.push({ label: traceId, value: traceId });
    }
  }
  for (const episode of episodes) {
    if (episode.trace_id && !seen.has(episode.trace_id)) {
      seen.add(episode.trace_id);
      options.push({ label: episode.trace_id, value: episode.trace_id });
    }
  }
  return options;
}

function firstSourceTrace(artifact?: ArtifactRecord): string {
  return String(artifact?.source_trace_ids?.[0] ?? "");
}

function toggleValue(values: string[], value: string): string[] {
  if (values.includes(value)) {
    return values.filter((item) => item !== value);
  }
  return [...values, value];
}
