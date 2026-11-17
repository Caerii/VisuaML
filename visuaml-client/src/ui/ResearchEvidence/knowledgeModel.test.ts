import { describe, expect, it } from 'vitest';
import { parseKnowledgeGraph, relatedKnowledge } from './knowledgeModel';

const h = (value: string) => value.repeat(64);
const revision = 'a'.repeat(40);
const source = 'benchmarks/pretrained_llama_native_cpu_benchmark_v28.json';
const older = 'benchmarks/native_online_attention_observation_v31.json';

const node = (id: string, artifact: string, path: string) => ({
  entry_id: h(id),
  kind: id === '1' ? 'pinned_pretrained_native_witness' : 'attention_context_sweep',
  label: id === '1' ? 'Pinned Tiny Llama' : 'Attention context',
  status: 'historical_finite_numeric_witness',
  evidence_class: 'historical_observation',
  source_path: path,
  artifact_sha256: h(artifact),
  workflow_url: 'https://github.com/Caerii/VisuaML/actions/runs/123',
  semantic_sha256: h('b'),
  generated_cpp_sha256: h('c'),
  native_binary_sha256: h('d'),
  checkpoint_sha256: h('e'),
  model_revision: revision,
  op_sequence: ['llama_decoder'],
  context: { model: 'manojredhat/tiny-llama' },
  result: { max_abs_error: 0.000001 },
  limitations: ['Finite witness, not a proof'],
  replay_plan_id: id === '1' ? h('f') : null,
});

const sample = () => ({
  schema: 'visuaml.knowledge-graph.v35',
  parent_catalog_sha256: h('1'),
  knowledge_graph_sha256: h('2'),
  reviewed_artifacts: {
    [source]: h('3'),
    [older]: h('4'),
  },
  nodes: [node('1', '3', source), node('5', '4', older)],
  edges: [
    {
      from: h('1'),
      to: h('5'),
      relation: 'identical_checkpoint_fingerprint',
      sha256: h('e'),
      scope: 'identity_only_not_transferable_correctness_or_performance',
    },
  ],
  plans: [
    {
      schema: 'visuaml.replay-plan.v35',
      plan_id: h('f'),
      source_artifact_path: source,
      source_artifact_sha256: h('3'),
      evidence_class: 'historical_ci_numerical_witness_not_replayed_here',
      model: 'manojredhat/tiny-llama',
      model_revision: revision,
      checkpoint_files_sha256: {
        'model.safetensors': h('e'),
        'config.json': h('6'),
      },
      source_files_sha256: { 'visuaml/compiler_lab/native_cpu_v28.cpp': h('c') },
      steps: ['verify_archive_bytes', 'verify_source_bytes'],
      historical_source_commit: null,
      status: 'preflight_only_no_executable_replay',
      limitations: ['Not a fresh numerical replay'],
      ci_run: 'https://github.com/Caerii/VisuaML/actions/runs/123',
    },
  ],
  counts: {
    studies: 2,
    source_artifacts: 2,
    identity_edges: 1,
    pinned_pretrained_witnesses: 1,
  },
  limitations: ['Identity not equivalence'],
});

describe('replayable computational knowledge graph', () => {
  it('preserves strict reviewed source and exact checkpoint lineage', () => {
    const graph = parseKnowledgeGraph(sample());
    expect(graph.nodes).toHaveLength(2);
    const edges = relatedKnowledge(graph, graph.nodes[0].entry_id);
    expect(edges).toHaveLength(1);
    expect(edges[0].node.label).toBe('Attention context');
    expect(edges[0].relation).toBe('identical_checkpoint_fingerprint');
  });

  it('rejects unapproved artifact paths, malformed IDs, and source drift', () => {
    expect(() =>
      parseKnowledgeGraph({
        ...sample(),
        nodes: [{ ...node('1', '3', '../../secret'), node: 'unknown' }, node('5', '4', older)],
      }),
    ).toThrow();
    expect(() =>
      parseKnowledgeGraph({
        ...sample(),
        nodes: [node('1', '7', source), node('5', '4', older)],
      }),
    ).toThrow();
    expect(() =>
      parseKnowledgeGraph({
        ...sample(),
        knowledge_graph_sha256: 'not-a-digest',
      }),
    ).toThrow();
  });

  it('rejects broken identity edges and missing study endpoints', () => {
    expect(() =>
      parseKnowledgeGraph({
        ...sample(),
        edges: [{ ...sample().edges[0], to: h('9') }],
      }),
    ).toThrow();
    expect(() =>
      parseKnowledgeGraph({
        ...sample(),
        edges: [{ ...sample().edges[0], scope: 'verified-universal-equivalence' }],
      }),
    ).toThrow();
    expect(() =>
      parseKnowledgeGraph({
        ...sample(),
        counts: { ...sample().counts, identity_edges: 2 },
      }),
    ).toThrow();
  });

  it('refuses to display a replay as executed or recovered source checkout', () => {
    const base = sample();
    expect(() =>
      parseKnowledgeGraph({
        ...base,
        plans: [{ ...base.plans[0], status: 'numerical_replayed_successfully' }],
      }),
    ).toThrow();
    expect(() =>
      parseKnowledgeGraph({
        ...base,
        plans: [{ ...base.plans[0], historical_source_commit: revision }],
      }),
    ).toThrow();
    expect(() =>
      parseKnowledgeGraph({
        ...base,
        plans: [
          {
            ...base.plans[0],
            checkpoint_files_sha256: {
              'config.json': 'not-a-hash',
            },
          },
        ],
      }),
    ).toThrow();
  });
});
