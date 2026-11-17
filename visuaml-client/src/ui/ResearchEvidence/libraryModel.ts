/** Versioned client projection of the Python-reviewed v34 evidence catalog.
 *
 * This checks structure and safe links; it does NOT authenticate the producer,
 * prove the mathematical claims or validate the underlying native binary.
 */

export type EvidenceEntry = {
  entry_id: string;
  artifact_sha256: string;
  source_path: string;
  schema: string;
  kind: string;
  label: string;
  evidence_class: string;
  status: string;
  workflow_url: string;
  source_commit: string | null;
  semantic_sha256: string | null;
  generated_cpp_sha256: string | null;
  native_binary_sha256: string | null;
  checkpoint_sha256: string | null;
  op_sequence: string[];
  context: Record<string, string>;
  result: Record<string, number>;
  limitations: string[];
};

export type EvidenceCatalog = {
  schema: 'visuaml.evidence-library.v34';
  entries: EvidenceEntry[];
  artifacts: Record<string, string>;
  catalog_sha256: string;
  counts: {
    studies: number;
    source_artifacts: number;
    kinds: Record<string, number>;
  };
  limitations: string[];
};

const sha256 = /^[0-9a-f]{64}$/;
const commitSha = /^[0-9a-f]{40}$/;
const runUrl = /^https:\/\/github\.com\/Caerii\/VisuaML\/actions\/runs\/[0-9]+$/;
const allowedPaths = new Set([
  'benchmarks/native_online_attention_observation_v31.json',
  'benchmarks/native_semantic_codegen_observation_v32.json',
  'benchmarks/export_evidence_observation_v33.json',
]);

const object = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const optionalHash = (value: unknown): boolean =>
  value === null || (typeof value === 'string' && sha256.test(value));

const stringArray = (value: unknown, maximum: number): value is string[] =>
  Array.isArray(value) &&
  value.length <= maximum &&
  value.every((part) => typeof part === 'string' && part.length <= 300);

const stringMap = (value: unknown, maximum: number): value is Record<string, string> =>
  object(value) &&
  Object.keys(value).length <= maximum &&
  Object.entries(value).every(
    ([key, item]) => key.length <= 48 && typeof item === 'string' && item.length <= 160,
  );

const numericalMap = (value: unknown): value is Record<string, number> =>
  object(value) &&
  Object.keys(value).length <= 20 &&
  Object.values(value).every(
    (item) => typeof item === 'number' && Number.isFinite(item) && item >= 0,
  );

export function parseEvidenceCatalog(value: unknown): EvidenceCatalog {
  if (
    !object(value) ||
    value.schema !== 'visuaml.evidence-library.v34' ||
    typeof value.catalog_sha256 !== 'string' ||
    !sha256.test(value.catalog_sha256) ||
    !Array.isArray(value.entries) ||
    !value.entries.length ||
    value.entries.length > 10000 ||
    !object(value.artifacts) ||
    Object.keys(value.artifacts).length > 256 ||
    !Object.entries(value.artifacts).every(
      ([path, digest]) =>
        allowedPaths.has(path) && typeof digest === 'string' && sha256.test(digest),
    ) ||
    !object(value.counts) ||
    (typeof value.counts.studies !== 'number' || !Number.isInteger(value.counts.studies)) ||
    (typeof value.counts.source_artifacts !== 'number' ||
      !Number.isInteger(value.counts.source_artifacts)) ||
    !numericalMap(value.counts.kinds) ||
    !stringArray(value.limitations, 30)
  ) {
    throw new Error('Malformed versioned research evidence catalog');
  }
  if (
    value.entries.length !== value.counts.studies ||
    Object.keys(value.artifacts).length !== value.counts.source_artifacts
  ) {
    throw new Error('Evidence catalog count and artifact identity disagree');
  }
  const seen = new Set<string>();
  for (const entry of value.entries) {
    if (
      !object(entry) ||
      typeof entry.entry_id !== 'string' ||
      !sha256.test(entry.entry_id) ||
      seen.has(entry.entry_id) ||
      typeof entry.artifact_sha256 !== 'string' ||
      !sha256.test(entry.artifact_sha256) ||
      typeof entry.source_path !== 'string' ||
      !allowedPaths.has(entry.source_path) ||
      value.artifacts[entry.source_path] !== entry.artifact_sha256 ||
      typeof entry.schema !== 'string' ||
      typeof entry.kind !== 'string' ||
      typeof entry.label !== 'string' ||
      typeof entry.evidence_class !== 'string' ||
      typeof entry.status !== 'string' ||
      typeof entry.workflow_url !== 'string' ||
      !runUrl.test(entry.workflow_url) ||
      !optionalHash(entry.semantic_sha256) ||
      !optionalHash(entry.generated_cpp_sha256) ||
      !optionalHash(entry.native_binary_sha256) ||
      !optionalHash(entry.checkpoint_sha256) ||
      !(
        entry.source_commit === null ||
        (typeof entry.source_commit === 'string' && commitSha.test(entry.source_commit))
      ) ||
      !stringArray(entry.op_sequence, 128) ||
      !stringMap(entry.context, 16) ||
      !numericalMap(entry.result) ||
      !stringArray(entry.limitations, 30)
    ) {
      throw new Error('Evidence entry violates its source-bound structural contract');
    }
    seen.add(entry.entry_id);
  }
  const actualKinds: Record<string, number> = {};
  for (const entry of value.entries) {
    const kind = (entry as EvidenceEntry).kind;
    actualKinds[kind] = (actualKinds[kind] || 0) + 1;
  }
  for (const key of new Set([
    ...Object.keys(actualKinds),
    ...Object.keys(value.counts.kinds),
  ])) {
    if (actualKinds[key] !== value.counts.kinds[key]) {
      throw new Error('Evidence category counts do not match the catalog');
    }
  }
  return value as EvidenceCatalog;
}

export function matchingEvidence(
  entries: EvidenceEntry[],
  query: { text?: string; kind?: string; operation?: string },
): EvidenceEntry[] {
  const phrase = (query.text || '').trim().toLowerCase();
  return entries.filter((entry) => {
    if (query.kind && query.kind !== 'all' && entry.kind !== query.kind) return false;
    if (query.operation && query.operation !== 'all' &&
        !entry.op_sequence.includes(query.operation)) return false;
    if (!phrase) return true;
    return [
      entry.label,
      entry.kind,
      entry.evidence_class,
      entry.status,
      ...entry.op_sequence,
      ...Object.values(entry.context),
    ].some((value) => value.toLowerCase().includes(phrase));
  });
}

export function exactLineage(entries: EvidenceEntry[], selected: EvidenceEntry): EvidenceEntry[] {
  // Binary identity is an observation, never a reason to skip new correctness tests.
  if (!selected.native_binary_sha256) return [];
  return entries.filter(
    (entry) =>
      entry.entry_id !== selected.entry_id &&
      entry.native_binary_sha256 === selected.native_binary_sha256,
  );
}
