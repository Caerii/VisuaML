/** Reviewable static CI snapshot. The schema is an inspection contract, not a signature. */

export type EvidenceModel = {
  family: string;
  kind: string;
  operations: string[];
  maximumAbsoluteError: number;
  secondInputMaximumAbsoluteError: number;
  semanticSha256: string;
  generatedCppSha256: string;
  binarySha256: string;
  compileMs: number;
  warmMedianMs: number;
};

export type EvidenceSnapshot = {
  schema: 'visuaml.public-evidence-snapshot.v1';
  asOf: string;
  sourceCommit: string;
  workflowRun: string;
  methodology: string;
  evidenceClass: string;
  models: EvidenceModel[];
  transformation: {
    rule: string;
    numericalStatus: 'passed_finite_witnesses' | 'rejected';
    evidenceSha256: string;
    experimentalStatus: string;
    medianRatioBeforeOverAfter: number;
    interpretation: string;
  };
  limitations: string[];
};

const hex64 = /^[0-9a-f]{64}$/;
const hex40 = /^[0-9a-f]{40}$/;
const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const positive = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;

export function parseEvidenceSnapshot(value: unknown): EvidenceSnapshot {
  if (
    !isRecord(value) ||
    value.schema !== 'visuaml.public-evidence-snapshot.v1' ||
    typeof value.asOf !== 'string' ||
    !/^\d{4}-\d{2}-\d{2}$/.test(value.asOf) ||
    typeof value.sourceCommit !== 'string' ||
    !hex40.test(value.sourceCommit) ||
    typeof value.workflowRun !== 'string' ||
    !/^https:\/\/github\.com\/Caerii\/VisuaML\/actions\/runs\/\d+$/.test(value.workflowRun) ||
    typeof value.methodology !== 'string' ||
    typeof value.evidenceClass !== 'string' ||
    !Array.isArray(value.models) ||
    value.models.length < 1 ||
    value.models.length > 8 ||
    !value.models.every(
      (model) =>
        isRecord(model) &&
        typeof model.family === 'string' &&
        typeof model.kind === 'string' &&
        Array.isArray(model.operations) &&
        model.operations.length > 0 &&
        model.operations.length <= 16 &&
        model.operations.every((op: unknown) => typeof op === 'string') &&
        positive(model.maximumAbsoluteError) &&
        positive(model.secondInputMaximumAbsoluteError) &&
        positive(model.compileMs) &&
        positive(model.warmMedianMs) &&
        typeof model.semanticSha256 === 'string' &&
        hex64.test(model.semanticSha256) &&
        typeof model.generatedCppSha256 === 'string' &&
        hex64.test(model.generatedCppSha256) &&
        typeof model.binarySha256 === 'string' &&
        hex64.test(model.binarySha256),
    ) ||
    !isRecord(value.transformation) ||
    typeof value.transformation.rule !== 'string' ||
    !['passed_finite_witnesses', 'rejected'].includes(
      String(value.transformation.numericalStatus),
    ) ||
    typeof value.transformation.evidenceSha256 !== 'string' ||
    !hex64.test(value.transformation.evidenceSha256) ||
    typeof value.transformation.experimentalStatus !== 'string' ||
    !positive(value.transformation.medianRatioBeforeOverAfter) ||
    typeof value.transformation.interpretation !== 'string' ||
    !Array.isArray(value.limitations) ||
    !value.limitations.every((item: unknown) => typeof item === 'string')
  ) {
    throw new Error('Unrecognized or invalid published research evidence snapshot');
  }
  // Narrowing the structural contract is local UI validation. It cannot
  // cryptographically authenticate the CI producer or independently rerun it.
  return value as EvidenceSnapshot;
}
