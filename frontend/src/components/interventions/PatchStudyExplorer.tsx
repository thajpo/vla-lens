import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchPatchStudies } from "../../api/interventions";
import type {
  PatchStudyAnalysis,
  PatchStudyCell,
  PatchStudyPair,
} from "../../types/interventions";

export function PatchStudyExplorer() {
  const query = useQuery({
    queryKey: ["patch-studies"],
    queryFn: fetchPatchStudies,
    staleTime: 15_000,
  });
  const studies = query.data?.patch_studies ?? [];
  const [selectedStudyId, setSelectedStudyId] = useState("");
  const study = studies.find((item) => item.study_id === selectedStudyId)
    ?? studies.find((item) => item.stream === "expert_action" && item.specificity.length)
    ?? studies[0];

  if (query.isLoading) {
    return <p className="app-message">Loading patch studies.</p>;
  }
  if (query.isError || !study) {
    return null;
  }
  return (
    <section className="patch-study-explorer">
      <header className="patch-study-header">
        <div>
          <span>Counterfactual patch study</span>
          <h2>{study.question || study.study_id}</h2>
        </div>
        <select
          aria-label="Patch study"
          value={study.study_id}
          onChange={(event) => setSelectedStudyId(event.target.value)}
        >
          {studies.map((item) => (
            <option key={item.study_id} value={item.study_id}>
              {studyLabel(item)}
            </option>
          ))}
        </select>
      </header>
      <p className="patch-study-lede">{studySummary(study)}</p>
      <div className="patch-study-meta">
        <span>{streamLabel(study)}</span>
        {study.stream === "expert_action" && <span>{generationStepLabel(study)}</span>}
        <span>{study.pair_count} matched scenes</span>
        <span>{study.planned_trial_count} patches</span>
        <span>{study.controls.length ? `${study.controls.length} controls` : "localization pass"}</span>
      </div>
      <PatchTransferMatrix study={study} />
      <PatchPairFrames pairs={study.pairs} />
    </section>
  );
}

function PatchTransferMatrix({ study }: { study: PatchStudyAnalysis }) {
  const cells = useMemo(
    () => new Map(study.summary.map((cell) => [`${cell.layer}:${cell.token_region}`, cell])),
    [study.summary],
  );
  return (
    <div className="patch-transfer-table-wrap">
      <table className="patch-transfer-table">
        <thead>
          <tr>
            <th>{study.stream === "expert_action" ? "Action layer" : "VLM layer"}</th>
            {study.token_regions.map((region) => <th key={region}>{regionLabel(region)}</th>)}
          </tr>
        </thead>
        <tbody>
          {study.layers.map((layer) => (
            <tr key={layer}>
              <th>{layer}</th>
              {study.token_regions.map((region) => {
                const cell = cells.get(`${layer}:${region}`);
                return <TransferCell cell={cell} key={region} />;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TransferCell({ cell }: { cell?: PatchStudyCell }) {
  if (!cell) return <td>—</td>;
  return (
    <td
      className={transferClass(cell.transfer_mean)}
      title={`95% interval ${formatPercent(cell.transfer_ci95_low)} to ${formatPercent(cell.transfer_ci95_high)}; ${cell.localized_pair_count}/${cell.pair_count} layouts passed the localization gates`}
    >
      <strong>{formatPercent(cell.transfer_mean)}</strong>
      <small>{formatPercent(cell.transfer_ci95_low)}–{formatPercent(cell.transfer_ci95_high)}</small>
    </td>
  );
}

function PatchPairFrames({ pairs }: { pairs: PatchStudyPair[] }) {
  const [selectedPairId, setSelectedPairId] = useState("");
  const pair = pairs.find((item) => item.pair_id === selectedPairId) ?? pairs[0];
  if (!pair) return null;
  return (
    <section className="patch-pair-inspector">
      <header>
        <div>
          <span>Matched scene</span>
          <strong>{objectLabel(pair.target_object, "Target")} ↔ {objectLabel(pair.distractor_object, "Distractor")}</strong>
        </div>
        <select
          aria-label="Counterfactual pair"
          value={pair.pair_id}
          onChange={(event) => setSelectedPairId(event.target.value)}
        >
          {pairs.map((item, index) => (
            <option key={item.pair_id} value={item.pair_id}>Layout {index}</option>
          ))}
        </select>
      </header>
      <div className="patch-pair-frames">
        <figure>
          <img alt="Original matched scene" src={frameUrl(pair.recipient_trace_id)} />
          <figcaption>Original</figcaption>
        </figure>
        <figure>
          <img alt="Scene with object poses exchanged" src={frameUrl(pair.donor_trace_id)} />
          <figcaption>Objects exchanged</figcaption>
        </figure>
      </div>
    </section>
  );
}

function studySummary(study: PatchStudyAnalysis): string {
  const best = study.headline;
  const bestText = best.best_transfer_mean == null
    ? "No completed transfer estimate."
    : `${regionLabel(String(best.best_token_region || "patch"))} at layer ${best.best_layer} transfers ${formatPercent(best.best_transfer_mean)} of the scene-driven action change.`;
  const failedSpecificity = study.specificity.filter((item) => !item.main_beats_control).length;
  if (study.specificity.length) {
    return `${bestText} Controls match or beat the intended patch in ${failedSpecificity}/${study.specificity.length} comparisons.`;
  }
  return bestText;
}

function studyLabel(study: PatchStudyAnalysis): string {
  const phase = study.specificity.length
    ? "Controls"
    : study.token_regions.some((region) => region !== "action_all")
      && study.stream === "expert_action"
      ? "Action positions"
      : "Layer sweep";
  const steps = study.stream === "expert_action" ? ` · ${shortGenerationStepLabel(study)}` : "";
  const scenes = `${study.pair_count} ${study.pair_count === 1 ? "scene" : "scenes"}`;
  return `${phase} · ${studyScopeLabel(study)}${steps} · ${scenes}`;
}

function streamLabel(study: PatchStudyAnalysis): string {
  return study.stream === "expert_action" ? "Action expert" : "Visual prefix";
}

function generationStepLabel(study: PatchStudyAnalysis): string {
  const steps = study.generation_steps;
  if (!steps || steps === "all") return "All 10 denoising steps";
  if (steps.indices?.length) {
    const joined = steps.indices.join(", ");
    return `Denoising steps ${joined}`;
  }
  if (steps.start != null && steps.end != null) {
    return `Denoising steps ${steps.start}–${steps.end - 1}`;
  }
  return "Selected denoising steps";
}

function shortGenerationStepLabel(study: PatchStudyAnalysis): string {
  const label = generationStepLabel(study);
  return label === "All 10 denoising steps"
    ? "all denoising steps"
    : label.toLowerCase();
}

function studyScopeLabel(study: PatchStudyAnalysis): string {
  if (study.stream === "expert_action") return "action expert";
  if (study.token_regions.includes("full_prefix")) return "broad visual scopes";
  return "object regions";
}

function regionLabel(region: string): string {
  return ({
    active_images: "Both cameras",
    action_all: "All 50 action positions",
    action_first_10: "First 10 actions",
    action_last_10: "Last 10 actions",
    action_middle_10: "Middle 10 actions",
    both: "Both objects",
    complement: "Background",
    distractor: "Mug",
    full_prefix: "Full prefix",
    language_active: "Language",
    main_camera: "Main camera",
    target: "Book",
    wrist_camera: "Wrist camera",
  } as Record<string, string>)[region] ?? region.replaceAll("_", " ");
}

function objectLabel(value: string | null | undefined, fallback: string): string {
  if (!value) return fallback;
  const words = value.replace(/_\d+$/, "").replaceAll("_", " ");
  if (words.includes("book")) return "Book";
  if (words.includes("mug")) return "Mug";
  return words.replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function frameUrl(traceId: string): string {
  return `/api/frame?trace_id=${encodeURIComponent(traceId)}&camera=main&timestep=0&source=auto`;
}

function transferClass(value: number): string {
  if (value >= 0.75) return "transfer-strong";
  if (value >= 0.1) return "transfer-partial";
  if (value < 0) return "transfer-negative";
  return "transfer-weak";
}

function formatPercent(value: number): string {
  return `${(100 * value).toFixed(Math.abs(value) < 0.1 ? 1 : 0)}%`;
}
