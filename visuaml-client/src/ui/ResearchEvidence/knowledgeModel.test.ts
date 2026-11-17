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

const freshSource = 'benchmarks/pretrained_llama_independent_replay_v36.json';

const replayedV36 = () => {
  const base = sample();
  const previous = base.nodes[0];
  const fresh = {
    ...node('6', '8', freshSource),
    kind: 'pinned_pretrained_independent_replay',
    label: 'Independent Tiny Llama v36',
    status: 'fresh_independent_numerical_witness_passed',
    source_commit: null,
    semantic_sha256: previous.semantic_sha256,
    checkpoint_sha256: previous.checkpoint_sha256,
    workflow_url: 'https://github.com/Caerii/VisuaML/actions/runs/44193012326',
  };
  return {
    ...base,
    schema: 'visuaml.knowledge-graph.v36',
    parent_graph_sha256: h('a'),
    reviewed_artifacts: { ...base.reviewed_artifacts, [freshSource]: h('8') },
    nodes: [...base.nodes, fresh],
    edges: [
      ...base.edges,
      {
        from: h('1'),
        to: h('6'),
        relation: 'identical_checkpoint_fingerprint',
        sha256: h('e'),
        scope: 'identity_only_not_transferable_correctness_or_performance',
      },
    ],
    counts: {
      studies: 3,
      source_artifacts: 3,
      identity_edges: 2,
      pinned_pretrained_witnesses: 2,
      distinct_pinned_pretrained_models: 1,
      fresh_independent_replays: 1,
    },
    fresh_replays: [
      {
        schema: 'visuaml.independent-pretrained-replay-summary.v36',
        study_id: h('6'),
        record_bytes_sha256: h('8'),
        numerical_record_sha256: h('9'),
        source_bundle_sha256: h('0'),
        source_file_commits: {
          'examples/native_cpu_bench_v28.py': revision,
          'visuaml/compiler_lab/native_cpu_v28.cpp': 'b'.repeat(40),
          'visuaml/compiler_lab/native_cpu_v28.py': 'c'.repeat(40),
        },
        producing_ci_run: 'https://github.com/Caerii/VisuaML/actions/runs/44193012326',
        producing_branch_head: revision,
        original_experiment_commit: null,
        numerical_replay_performed: true,
        passed: true,
        limits: 'Finite real-model numerical test; not formal proof.',
      },
    ],
  };
};

describe('v36 fresh pretrained numerical replay integrity', () => {
  it('admits new source-bound evidence without mutating the original preflight', () => {
    const graph = parseKnowledgeGraph(replayedV36());
    expect(graph.schema).toBe('visuaml.knowledge-graph.v36');
    expect(graph.fresh_replays?.[0].numerical_replay_performed).toBe(true);
    expect(graph.plans[0].status).toBe('preflight_only_no_executable_replay');
    expect(graph.counts.distinct_pinned_pretrained_models).toBe(1);
    expect(relatedKnowledge(graph, h('6'))).toHaveLength(1);
  });

  it('rejects false success, changed archive and invented original commit', () => {
    const graph = replayedV36();
    expect(() =>
      parseKnowledgeGraph({
        ...graph,
        fresh_replays: [{ ...graph.fresh_replays[0], passed: false }],
      }),
    ).toThrow();
    expect(() =>
      parseKnowledgeGraph({
        ...graph,
        fresh_replays: [{ ...graph.fresh_replays[0], original_experiment_commit: revision }],
      }),
    ).toThrow();
    expect(() =>
      parseKnowledgeGraph({
        ...graph,
        nodes: graph.nodes.map((item) =>
          item.entry_id === h('6') ? { ...item, artifact_sha256: h('f') } : item,
        ),
      }),
    ).toThrow();
    expect(() =>
      parseKnowledgeGraph({
        ...graph,
        fresh_replays: [
          {
            ...graph.fresh_replays[0],
            source_file_commits: { '../untracked': revision },
          },
        ],
      }),
    ).toThrow();
  });

  it('rejects a replay falsely presented as a second pretrained model family', () => {
    const graph = replayedV36();
    expect(() =>
      parseKnowledgeGraph({
        ...graph,
        counts: { ...graph.counts, distinct_pinned_pretrained_models: 2 },
      }),
    ).toThrow();
  });
});
