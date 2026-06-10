export const researchCopy = {
  labels: {
    activationSources: "activation sources",
    artifactReadout: "Probe readout",
    datasetDelta: "Dataset delta",
    datasetScore: "Dataset score",
    episodeOrder: "episode order",
    inputFeature: "Input feature",
    probeShortlist: "Probe shortlist",
    readSource: "Read source",
    refineEpisodes: "Refine episodes",
    resultCoverage: "Probe results",
    splitCoverage: "Split coverage",
  },
  probeVerdict: {
    noScores: {
      detail: "No compatible probe scores for this dataset view.",
      headline: "No scored evidence yet",
      label: "Unknown",
    },
    trainOnly: {
      detail: "Only training episodes are scored. Add validation or test episodes before treating this as evidence.",
      headline: "Train evidence only",
      label: "Training only",
    },
    reviewFailures: {
      detail: "Best use: review suspicious failures. Trust: inspect episodes before drawing conclusions.",
      headline: "Good review lens",
      label: "Needs review",
    },
    heldoutAvailable: {
      detail: "Held-out episodes are scored and high-confidence failures are not dominating this view.",
      headline: "Held-out evidence available",
      label: "Usable",
    },
  },
  unavailable: {
    episodeLens: "This probe is selected, but no episode-level evidence is available yet.",
    probeReadout: "No probe readout for this episode.",
    rawActivationFallback: "The probe evidence request failed, so the inspector is showing the raw activation view.",
  },
} as const;

