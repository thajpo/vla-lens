import type { ActivationSite, ArchitectureMetadata } from "../../types/dataset";
import type {
  PipelineDiagramArrow,
  PipelineDiagramBand,
  PipelineDiagramLayout,
  PipelineDiagramNode,
  PipelineDiagramPort,
  PipelineFamily,
  PipelineNode,
  PipelineSiteChoice,
  PipelineStage,
} from "./shared";
import {
  captureGroupForSite,
  filterVisibleCaptureSites,
  inspectionModeForSite,
  isInspectableSite,
  siteOptionLabel,
  sortSites,
  summarizeSiteKinds,
} from "./siteModel";

export function modelPipelineStages(sites: ActivationSite[]): PipelineStage[] {
  if (!sites.length) {
    return [];
  }
  if (!sites.some((site) => site.name.startsWith("pi05."))) {
    return fallbackPipelineStages(sites);
  }

  const prefixSites = sites.filter(
    (site) =>
      site.name === "pi05.vlm.prefix.image_hidden_tokens" ||
      site.name === "pi05.vlm.prefix.input_embeddings",
  );
  const vlmLayerNodes = Array.from({ length: 18 }, (_, layer) =>
    pipelineLayerNode("vlm", layer, sites.filter((site) => site.name.includes(`.vlm.layers.${layer}.`))),
  );
  const generationSites = sites.filter(
    (site) =>
      site.name === "pi05.expert.by_step.input_embeddings" ||
      site.name === "pi05.expert.by_step.position_ids" ||
      site.name === "pi05.expert.by_step.attention_mask" ||
      site.name === "pi05.expert.by_step.causal_mask" ||
      site.name === "pi05.expert.by_step.rope.cos" ||
      site.name === "pi05.expert.by_step.rope.sin",
  );
  const expertLayerNodes = Array.from({ length: 18 }, (_, layer) =>
    pipelineLayerNode("expert", layer, sites.filter((site) => site.name.includes(`.expert.layers.${layer}.`))),
  );
  const actionHeadSites = sites.filter(
    (site) => site.name.includes("pi05.action_head") && site.role !== "action_head_output",
  );
  const actionOutputSites = sites.filter(
    (site) =>
      site.name.includes("action_chunk") ||
      site.name.includes("action_output") ||
      site.role === "action_head_output",
  );

  const stages: PipelineStage[] = [];
  stages.push({
    id: "prefix",
    label: "Input Embed",
    family: "input",
    nodes: [pipelineNode("prefix-embed", "Inputs", "prompt + images", "input", prefixSites)],
  });
  stages.push({
    id: "vlm",
    label: "VLM",
    family: "vlm",
    nodes: vlmLayerNodes,
  });
  stages.push({
    id: "generation",
    label: "Denoising State",
    family: "action",
    nodes: [pipelineNode("generation-step", "x_t", "current action state", "action", generationSites)],
  });
  stages.push({
    id: "expert",
    label: "Action Expert",
    family: "expert",
    nodes: expertLayerNodes,
  });
  stages.push({
    id: "action-head",
    label: "Action Head",
    family: "action",
    nodes: [
      pipelineNode("action-head", "Head", "projection", "action", actionHeadSites),
      pipelineNode("action-output", "Action", "final chunk", "action", actionOutputSites),
    ],
  });
  return stages;
}

export function pipelineDiagramLayout(
  stages: PipelineStage[],
  architecture?: ArchitectureMetadata,
): PipelineDiagramLayout {
  const stage = (id: string) => stages.find((entry) => entry.id === id)?.nodes ?? [];
  const prefixNodes = stage("prefix");
  const vlmNodes = stage("vlm");
  const generationNodes = stage("generation");
  const expertNodes = stage("expert");
  const actionHeadNodes = stage("action-head");
  const isPi05Layout = prefixNodes.length > 0 || expertNodes.length > 0;
  const nodes: PipelineDiagramNode[] = [];
  const bands: PipelineDiagramBand[] = [];

  if (!isPi05Layout) {
    let cursor = 0;
    for (const entry of stages) {
      for (const node of entry.nodes) {
        nodes.push({ height: 62, node, stageId: entry.id, width: 92, x: cursor, y: 84 });
        cursor += 104;
      }
      cursor += 36;
    }
    return {
      arrows: sequentialArrows(nodes, "fallback"),
      bands: nodes.length ? [{ className: "other", height: 108, id: "captured", label: "Captured Sites", width: cursor, x: 0, y: 58 }] : [],
      height: 230,
      nodes,
      ports: [],
      width: Math.max(860, cursor + 40),
    };
  }

  const layerWidth = 24;
  const layerHeight = 82;
  const layerStep = 34;
  const yPrefix = 78;
  const yExpert = 254;
  const prefix = prefixNodes[0];
  const currentAction = generationNodes[0];
  const actionHead = actionHeadNodes[0];
  const actionOutput = actionHeadNodes[1];
  const vlmStart = 166;
  const expertStart = vlmStart;
  const vlmEnd = vlmStart + Math.max(0, vlmNodes.length - 1) * layerStep + layerWidth;
  const expertEnd = expertStart + Math.max(0, expertNodes.length - 1) * layerStep + layerWidth;
  const kvBusY = 196;
  const headX = expertEnd + 48;
  const outputX = headX + 118;

  const add = (
    node: PipelineNode | undefined,
    stageId: string,
    x: number,
    y: number,
    width: number,
    height: number,
  ) => {
    if (!node) {
      return;
    }
    nodes.push({ height, node, stageId, width, x, y });
  };

  add(prefix, "prefix", 42, yPrefix + 9, 92, 64);
  vlmNodes.forEach((node, index) => add(node, "vlm", vlmStart + index * layerStep, yPrefix, layerWidth, layerHeight));
  add(currentAction, "generation", 42, yExpert + 9, 92, 64);
  expertNodes.forEach((node, index) => add(node, "expert", expertStart + index * layerStep, yExpert, layerWidth, layerHeight));
  add(actionHead, "action-head", headX, yExpert + 9, 96, 64);
  add(actionOutput, "action-head", outputX, yExpert + 9, 104, 64);

  bands.push(
    { className: "vlm", height: 116, id: "vlm-band", label: "Vision-language input pass", width: vlmEnd + 36, x: 26, y: 48 },
    { className: "expert", height: 134, id: "expert-band", label: "Action denoiser", width: outputX + 132 - 26, x: 26, y: 224 },
  );

  const byId = (id: string) => nodes.find((entry) => entry.node.id === id);
  const firstVlm = vlmNodes[0] ? byId(vlmNodes[0].id) : undefined;
  const lastVlm = vlmNodes.at(-1) ? byId(vlmNodes.at(-1)!.id) : undefined;
  const firstExpert = expertNodes[0] ? byId(expertNodes[0].id) : undefined;
  const lastExpert = expertNodes.at(-1) ? byId(expertNodes.at(-1)!.id) : undefined;
  const prefixBox = prefix ? byId(prefix.id) : undefined;
  const currentBox = currentAction ? byId(currentAction.id) : undefined;
  const headBox = actionHead ? byId(actionHead.id) : undefined;
  const outputBox = actionOutput ? byId(actionOutput.id) : undefined;
  const arrows: PipelineDiagramArrow[] = [];
  const ports: PipelineDiagramPort[] = [];
  if (prefixBox && firstVlm) {
    const y = centerY(prefixBox);
    arrows.push({
      className: "forward",
      id: "prefix-to-vlm",
      path: `M ${right(prefixBox) + 4} ${y} L ${left(firstVlm) - 8} ${y}`,
    });
  }
  if (vlmNodes.length && expertNodes.length) {
    const kvLayers = perLayerKvEdgeLayers(architecture);
    const kvEndpointLayers = kvLayers.length
      ? uniqueNumbers([kvLayers[0], kvLayers[kvLayers.length - 1]])
      : [];
    const pairEndpoints = kvEndpointLayers.length
      ? kvEndpointLayers.map((layer) => byId(`vlm-${layer}`)).filter(isDiagramNode)
      : uniqueDiagramNodes([firstVlm, lastVlm]);
    pairEndpoints.forEach((vlm) => {
      const expert = byId(vlm.node.id.replace("vlm", "expert"));
      if (!expert) {
        return;
      }
      const pairX = centerX(vlm);
      arrows.push({
        className: "conditioning kv-pair",
        id: `kv-pair-${vlm.node.id}`,
        path: `M ${pairX} ${bottom(vlm) + 4} V ${top(expert) - 8}`,
      });
    });
    if (firstVlm && lastVlm) {
      ports.push({
        className: "kv-label",
        id: "kv-same-index-label",
        label: "same-index prefix memory",
        textAnchor: "middle",
        x: (centerX(firstVlm) + centerX(lastVlm)) / 2,
        y: kvBusY - 16,
      });
    }
    const dotLayers = kvLayers.length
      ? kvLayers.filter((layer) => !kvEndpointLayers.includes(layer))
      : [4, 8, 12];
    const dotLabelIndex = Math.floor(dotLayers.length / 2);
    dotLayers.forEach((layer, index) => {
      const vlm = byId(`vlm-${layer}`);
      if (!vlm) {
        return;
      }
      ports.push({
        className: "kv-dot",
        id: `kv-dot-${layer}`,
        label: index === dotLabelIndex ? "..." : undefined,
        radius: 2.6,
        textAnchor: "middle",
        x: centerX(vlm),
        y: kvBusY,
      });
    });
  }
  if (currentBox && firstExpert) {
    arrows.push({
      className: "forward",
      id: "current-action-to-expert",
      path: `M ${right(currentBox) + 4} ${centerY(currentBox)} L ${left(firstExpert) - 8} ${centerY(firstExpert)}`,
    });
  }
  if (lastExpert && headBox) {
    arrows.push({
      className: "forward",
      id: "expert-to-head",
      path: `M ${right(lastExpert) + 8} ${centerY(lastExpert)} L ${left(headBox) - 8} ${centerY(headBox)}`,
    });
  }
  if (headBox && outputBox) {
    arrows.push({
      className: "forward",
      id: "head-to-output",
      path: `M ${right(headBox) + 6} ${centerY(headBox)} L ${left(outputBox) - 8} ${centerY(outputBox)}`,
    });
  }
  if (currentBox && outputBox) {
    const loopY = bottom(outputBox) + 44;
    const returnX = centerX(currentBox);
    const outputXCenter = centerX(outputBox);
    arrows.push({
      className: "loop",
      id: "denoise-update-loop",
      label: "Euler update: x_t + dt * v_t",
      labelAnchor: "middle",
      labelX: (returnX + outputXCenter) / 2,
      labelY: loopY + 19,
      path: `M ${outputXCenter} ${bottom(outputBox) + 4} C ${outputXCenter} ${loopY - 16}, ${outputXCenter - 18} ${loopY}, ${outputXCenter - 42} ${loopY} H ${returnX + 42} C ${returnX + 18} ${loopY}, ${returnX} ${loopY - 16}, ${returnX} ${bottom(currentBox) + 6}`,
    });
  }
  if (outputBox) {
    arrows.push({
      className: "final",
      id: "final-action",
      path: `M ${right(outputBox) + 6} ${centerY(outputBox)} L ${right(outputBox) + 74} ${centerY(outputBox)}`,
    });
  }

  return {
    arrows,
    bands,
    height: 408,
    nodes,
    ports,
    width: Math.max(1080, outputX + 190),
  };
}

export function perLayerKvEdgeLayers(architecture?: ArchitectureMetadata): number[] {
  const layers = architecture?.edges
    ?.filter((edge) => edge.kind === "per_layer_kv_conditioning")
    .map((edge) => Number(edge.layer))
    .filter((layer) => Number.isInteger(layer)) ?? [];
  return uniqueNumbers(layers).sort((left, right) => left - right);
}

export function uniqueNumbers(values: number[]): number[] {
  return Array.from(new Set(values));
}

export function isDiagramNode(value: PipelineDiagramNode | undefined): value is PipelineDiagramNode {
  return Boolean(value);
}

export function uniqueDiagramNodes(
  entries: Array<PipelineDiagramNode | undefined>,
): PipelineDiagramNode[] {
  const seen = new Set<string>();
  const nodes: PipelineDiagramNode[] = [];
  for (const entry of entries) {
    if (!entry || seen.has(entry.node.id)) {
      continue;
    }
    seen.add(entry.node.id);
    nodes.push(entry);
  }
  return nodes;
}

export function sequentialArrows(nodes: PipelineDiagramNode[], prefix: string): PipelineDiagramArrow[] {
  return nodes.slice(0, -1).map((node, index) => {
    const next = nodes[index + 1];
    return {
      className: "forward",
      id: `${prefix}-${node.node.id}-${next.node.id}`,
      path: `M ${right(node)} ${centerY(node)} L ${left(next)} ${centerY(next)}`,
    };
  });
}

export function left(box: PipelineDiagramNode): number {
  return box.x;
}

export function right(box: PipelineDiagramNode): number {
  return box.x + box.width;
}

export function top(box: PipelineDiagramNode): number {
  return box.y;
}

export function bottom(box: PipelineDiagramNode): number {
  return box.y + box.height;
}

export function centerX(box: PipelineDiagramNode): number {
  return box.x + box.width / 2;
}

export function centerY(box: PipelineDiagramNode): number {
  return box.y + box.height / 2;
}

export function fallbackPipelineStages(sites: ActivationSite[]): PipelineStage[] {
  const byModule = new Map<string, ActivationSite[]>();
  for (const site of sites) {
    const key = site.module || site.segment || site.family || "model";
    byModule.set(key, [...(byModule.get(key) ?? []), site]);
  }
  return [
    {
      id: "captured-sites",
      label: "Captured Sites",
      family: "other",
      nodes: [...byModule.entries()].map(([module, moduleSites]) =>
        pipelineNode(
          `site-${module}`,
          module.split(".").at(-1) || "Site",
          summarizeSiteKinds(moduleSites),
          "other",
          moduleSites,
        ),
      ),
    },
  ];
}

export function pipelineLayerNode(
  family: Extract<PipelineFamily, "vlm" | "expert">,
  layer: number,
  sites: ActivationSite[],
): PipelineNode {
  return pipelineNode(`${family}-${layer}`, `L${layer}`, summarizeSiteKinds(sites), family, sites);
}

export function pipelineNode(
  id: string,
  label: string,
  fallbackSublabel: string,
  family: PipelineFamily,
  sites: ActivationSite[],
): PipelineNode {
  const inspectableSites = sortSites(sites.filter(isInspectableSite));
  const sortedSites = sortSites(filterVisibleCaptureSites(inspectableSites));
  const toChoice = (site: ActivationSite): PipelineSiteChoice => ({
    group: captureGroupForSite(site),
    id: site.site_id || site.name,
    label: siteOptionLabel(site),
    mode: inspectionModeForSite(site),
    site,
  });
  return {
    id,
    label,
    sublabel: sortedSites.length ? summarizeSiteKinds(sortedSites) : fallbackSublabel,
    family,
    captured: sortedSites.length > 0,
    sites: sortedSites,
    allSites: inspectableSites,
    choices: sortedSites.map(toChoice),
    rawChoices: inspectableSites.map(toChoice),
  };
}
