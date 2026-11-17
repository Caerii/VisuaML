/** Purely structural validation for a historical, source-bound knowledge graph.
 * A valid JSON shape does not authenticate CI authorship or prove numerical parity.
 */

export type KnowledgeNode = {
  entry_id: string;
  kind: string;
  label: string;
  status: string;
  evidence_class: string;
  source_path: string;
  artifact_sha256: string;
  workflow_url: string;
  semantic_sha256: string | null;
  native_binary_sha256: string | null;
  checkpoint_sha256: string | null;
  generated_cpp_sha256: string | null;
  model_revision?: string | null;
  op_sequence: string[];
  result: Record<string, number>;
  context: Record<string, string>;
  limitations: string[];
  replay_plan_id?: string | null;
};

export type KnowledgeEdge = {
  from: string;
  to: string;
  relation: string;
  sha256: string;
  scope: 'identity_only_not_transferable_correctness_or_performance';
};

export type ReplayPlan = {
  schema: 'visuaml.replay-plan.v35';
  plan_id: string;
  source_artifact_path: string;
  source_artifact_sha256: string;
  evidence_class: string;
  model: string;
  model_revision: string;
  checkpoint_files_sha256: Record<string, string>;
  source_files_sha256: Record<string, string>;
  steps: string[];
  historical_source_commit: null;
  status: 'preflight_only_no_executable_replay';
  limitations: string[];
  ci_run: string;
};

export type KnowledgeGraph = {
  schema: 'visuaml.knowledge-graph.v35';
  parent_catalog_sha256: string;
  knowledge_graph_sha256: string;
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  plans: ReplayPlan[];
  reviewed_artifacts: Record<string, string>;
  counts: {
    studies: number;
    source_artifacts: number;
    identity_edges: number;
    pinned_pretrained_witnesses: number;
  };
  limitations: string[];
};

const sha = /^[0-9a-f]{64}$/;
const revision = /^[0-9a-f]{40}$/;
const githubRun = /^https:\/\/github\.com\/Caerii\/VisuaML\/actions\/runs\/[0-9]+$/;
const paths = new Set([
  'benchmarks/native_online_attention_observation_v31.json',
  'benchmarks/native_semantic_codegen_observation_v32.json',
  'benchmarks/export_evidence_observation_v33.json',
  'benchmarks/pretrained_llama_native_cpu_benchmark_v28.json',
]);
const relationships = new Set([
  'identical_native_binary_bytes',
  'identical_generated_cpp_bytes',
  'identical_semantic_digest',
  'identical_checkpoint_fingerprint',
]);
const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const digest = (value: unknown): value is string =>
  typeof value === 'string' && sha.test(value);
const optionalDigest = (value: unknown): boolean => value === null || digest(value);
const names = (value: unknown): value is string[] =>
  Array.isArray(value) && value.length <= 128 &&
  value.every((item) => typeof item === 'string' && item.length <= 512);
const strings = (value: unknown): value is Record<string, string> =>
  isObject(value) && Object.keys(value).length <= 24 &&
  Object.entries(value).every(([key, item]) =>
    key.length <= 100 && typeof item === 'string' && item.length <= 512);
const numbers = (value: unknown): value is Record<string, number> =>
  isObject(value) && Object.keys(value).length <= 32 &&
  Object.values(value).every(
    (item) => typeof item === 'number' && Number.isFinite(item) && item >= 0,
  );

export function parseKnowledgeGraph(value: unknown): KnowledgeGraph {
  if (!isObject(value) ||
      value.schema !== 'visuaml.knowledge-graph.v35' ||
      !digest(value.parent_catalog_sha256) ||
      !digest(value.knowledge_graph_sha256) ||
      !isObject(value.counts) ||
      (typeof value.counts.studies !== 'number' ||
        !Number.isInteger(value.counts.studies)) ||
      (typeof value.counts.source_artifacts !== 'number' ||
        !Number.isInteger(value.counts.source_artifacts)) ||
      (typeof value.counts.identity_edges !== 'number' ||
        !Number.isInteger(value.counts.identity_edges)) ||
      value.counts.pinned_pretrained_witnesses !== 1 ||
      !isObject(value.reviewed_artifacts) ||
      !Object.entries(value.reviewed_artifacts).every(
        ([path, hash]) => paths.has(path) && digest(hash)) ||
      !Array.isArray(value.nodes) || value.nodes.length < 1 || value.nodes.length > 10000 ||
      !Array.isArray(value.edges) || value.edges.length > 100000 ||
      !Array.isArray(value.plans) || value.plans.length !== 1 ||
      !names(value.limitations)) {
    throw new Error('Invalid versioned computational knowledge graph');
  }

  if (value.nodes.length !== value.counts.studies ||
      value.edges.length !== value.counts.identity_edges ||
      Object.keys(value.reviewed_artifacts).length !== value.counts.source_artifacts) {
    throw new Error('Knowledge graph counts do not match reviewed source records');
  }
  const ids = new Set<string>();
  for (const node of value.nodes) {
    if (!isObject(node) ||
        !digest(node.entry_id) || ids.has(node.entry_id) ||
        typeof node.label !== 'string' || typeof node.kind !== 'string' ||
        typeof node.status !== 'string' || typeof node.evidence_class !== 'string' ||
        !paths.has(String(node.source_path)) ||
        !digest(node.artifact_sha256) ||
        value.reviewed_artifacts[String(node.source_path)] !== node.artifact_sha256 ||
        typeof node.workflow_url !== 'string' || !githubRun.test(node.workflow_url) ||
        !optionalDigest(node.semantic_sha256) ||
        !optionalDigest(node.generated_cpp_sha256) ||
        !optionalDigest(node.native_binary_sha256) ||
        !optionalDigest(node.checkpoint_sha256) ||
        (node.model_revision !== undefined && node.model_revision !== null &&
          (typeof node.model_revision !== 'string' || node.model_revision.length > 160)) ||
        !names(node.op_sequence) || !strings(node.context) ||
        !numbers(node.result) || !names(node.limitations) ||
        (node.replay_plan_id !== undefined && !optionalDigest(node.replay_plan_id))) {
      throw new Error('Invalid source-bound knowledge node');
    }
    ids.add(node.entry_id as string);
  }
  for (const edge of value.edges) {
    if (!isObject(edge) || !ids.has(String(edge.from)) || !ids.has(String(edge.to)) ||
        edge.from === edge.to || !relationships.has(String(edge.relation)) ||
        !digest(edge.sha256) ||
        edge.scope !== 'identity_only_not_transferable_correctness_or_performance') {
      throw new Error('Invalid exact-identity knowledge graph edge');
    }
  }
  const plan = value.plans[0];
  if (!isObject(plan) || plan.schema !== 'visuaml.replay-plan.v35' ||
      !digest(plan.plan_id) || !digest(plan.source_artifact_sha256) ||
      !paths.has(String(plan.source_artifact_path)) ||
      value.reviewed_artifacts[String(plan.source_artifact_path)] !== plan.source_artifact_sha256 ||
      !strings(plan.checkpoint_files_sha256) || !strings(plan.source_files_sha256) ||
      !Object.values(plan.checkpoint_files_sha256).every(digest) ||
      !Object.values(plan.source_files_sha256).every(digest) ||
      typeof plan.model !== 'string' || typeof plan.model_revision !== 'string' ||
      !revision.test(plan.model_revision) ||
      plan.historical_source_commit !== null ||
      plan.status !== 'preflight_only_no_executable_replay' ||
      !names(plan.steps) || !names(plan.limitations) ||
      typeof plan.ci_run !== 'string' || !githubRun.test(plan.ci_run)) {
    throw new Error('Replay plan violates strict historical admission');
  }
  return value as KnowledgeGraph;
}

export function relatedKnowledge(
  graph: KnowledgeGraph,
  selectedId: string,
): { node: KnowledgeNode; relation: string }[] {
  const indexed = new Map(graph.nodes.map((node) => [node.entry_id, node]));
  return graph.edges.flatMap((edge) => {
    if (edge.from !== selectedId && edge.to !== selectedId) return [];
    const other = indexed.get(edge.from === selectedId ? edge.to : edge.from);
    return other ? [{ node: other, relation: edge.relation }] : [];
  });
}
