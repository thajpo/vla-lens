export const researchCopy = {
  labels: {
    activationSources: "activation tensors",
    artifactReadout: "Probe result",
    datasetDelta: "Score lift",
    datasetScore: "Best dataset score",
    episodeOrder: "episode order",
    inputFeature: "Input feature",
    probeShortlist: "Probe shortlist",
    readSource: "Probe input",
    refineEpisodes: "Refine episodes",
    resultCoverage: "Probe results",
    splitCoverage: "Split coverage",
  },
  probeVerdict: {
    noScores: {
      detail: "No probe scores for episodes in this view.",
      headline: "No scored episodes yet",
      label: "Unknown",
    },
    trainOnly: {
      detail: "Only training episodes are scored. Validation or test scores are required for generalization.",
      headline: "Training split only",
      label: "Training only",
    },
    reviewFailures: {
      detail: "Review high-confidence errors before trusting this probe.",
      headline: "Review failures",
      label: "Needs review",
    },
    heldoutAvailable: {
      detail: "Held-out episodes are scored and high-confidence failures are not dominating this view.",
      headline: "Validation/test scores available",
      label: "Usable",
    },
  },
  unavailable: {
    episodeLens: "No episode result is available for this probe yet.",
    probeReadout: "No probe result for this episode.",
    rawActivationFallback: "Probe result failed. Showing activation traces instead.",
  },
} as const;
