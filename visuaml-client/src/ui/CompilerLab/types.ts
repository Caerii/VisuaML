/** API declarations are deliberately narrow and versioned; no arbitrary Python code inputs. */
export type Scenario = 'mlp' | 'attention' | 'assembly' | 'convolution';
export type TensorSpec = { shape: string[]; dtype: string; bytes: number | null };
export type CompilerNode = {
  id: string;
  data: {
    label: string;
    name: string;
    op: string;
    target: string;
    outputShape: string | null;
    estimatedFlops: number | null;
    tensors: TensorSpec[];
    backend: string | null;
    loweringReason: string | null;
  };
  position: { x: number; y: number };
};
export type CompilerEdge = { id: string; source: string; target: string };
export type CostReport = {
  graph_digest: string;
  node_count: number;
  known_flops: number;
  peak_live_activation_bytes: number;
  parameter_bytes: number;
  uncounted_flop_nodes: string[];
  warnings: string[];
};
export type Stage = {
  name: string;
  graph_sha256: string;
  cost: CostReport;
  canvas: { nodes: CompilerNode[]; edges: CompilerEdge[] };
};
export type PassReceipt = {
  pass_name: string;
  before_sha256: string;
  after_sha256: string;
  before_nodes: number;
  after_nodes: number;
  witnesses_checked: number;
  max_abs_error: number;
  passed: boolean;
};
export type Assignment = {
  node: string;
  op: string;
  assigned: string;
  host_cost: number;
  device_cost: number | null;
};
export type Placement = {
  schema: string;
  profile: string;
  backend: string;
  estimated_total_cost: number;
  all_host_baseline: number;
  savings_vs_all_host: number;
  assignments: Assignment[];
  boundaries: { producer: string; consumer: string; bytes: number; estimated_cost: number }[];
  warnings: string[];
};
export type PlacementRobustness = {
  schema: string;
  search: string;
  candidates_checked: number;
  scenarios: { profile: string; optimum_cost: number; device_nodes: string[] }[];
  preferred: {
    device_nodes: string[];
    worst_regret: number;
    worst_cost: number;
    mean_regret: number;
  };
  node_device_frequency: [string, number][];
  warnings: string[];
};

export type CompilerReport = {
  schema: string;
  scenario: Scenario;
  seed: number;
  api_contract: { schema: string; source: string; result_kind: string; backend_execution: string };
  stages: Stage[];
  passes: PassReceipt[];
  witness: {
    passed: boolean;
    leaves: { path: string; dtype: string; max_abs_error: number; passed: boolean }[];
  };
  torch_export_roundtrip: { passed: boolean };
  hypothetical_partition_sweep: { transfer_overhead: number; partition: Placement }[];
  hypothetical_placement_robustness?: PlacementRobustness;
  limitations: string[];
};

export async function analyzeScenario(scenario: Scenario, seed = 19): Promise<CompilerReport> {
  const response = await fetch('/api/compiler-lab/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario, seed }),
  });
  const json: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message =
      json && typeof json === 'object' && 'message' in json
        ? String(json.message)
        : response.statusText;
    throw new Error(message);
  }
  if (!json || typeof json !== 'object' || !('api_contract' in json) || !('stages' in json)) {
    throw new Error('Invalid Compiler Lab API response');
  }
  const value = json as CompilerReport;
  if (
    value.api_contract?.schema !== 'visuaml.compiler-lab.api.v1' ||
    !Array.isArray(value.stages)
  ) {
    throw new Error('Unsupported Compiler Lab API contract');
  }
  return value;
}
