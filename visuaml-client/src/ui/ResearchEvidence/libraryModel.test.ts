import { describe, expect, it } from 'vitest';
import {
  exactLineage,
  matchingEvidence,
  parseEvidenceCatalog,
  type EvidenceEntry,
} from './libraryModel';

const h = (character: string) => character.repeat(64);
const path = 'benchmarks/export_evidence_observation_v33.json';

const model = (id: string, kind = 'native_exported_graph'): EvidenceEntry => ({
  entry_id: h(id),
  artifact_sha256: h('a'),
  source_path: path,
  schema: 'visuaml.export-evidence-demonstration.v33',
  kind,
  label: kind === 'rewrite_evaluation' ? 'relu-idempotence-v1' : 'Residual MLP',
  evidence_class: 'source_bound_finite_numeric_witnesses',
  status: 'finite_numeric_witnesses',
  workflow_url: 'https://github.com/Caerii/VisuaML/actions/runs/1234',
  source_commit: 'b'.repeat(40),
  semantic_sha256: h('c'),
  generated_cpp_sha256: h('d'),
  native_binary_sha256: h('e'),
  checkpoint_sha256: null,
  op_sequence: ['matmul', 'relu'],
  context: { frontend: 'trusted_static_torch_export', model: 'synthetic' },
  result: { max_abs_error_eager_torch: 0 },
  limitations: ['No formal equivalence proof'],
});

const fixture = () => ({
  schema: 'visuaml.evidence-library.v34',
  entries: [model('1'), model('2', 'rewrite_evaluation')],
  artifacts: { [path]: h('a') },
  catalog_sha256: h('f'),
  counts: {
    studies: 2,
    source_artifacts: 1,
    kinds: { native_exported_graph: 1, rewrite_evaluation: 1 },
  },
  limitations: ['No automatic promotion'],
});

describe('source-bound evidence catalog', () => {
  it('admits a finite reviewed catalog with exact source identity', () => {
    const entries = parseEvidenceCatalog(fixture()).entries;
    expect(entries).toHaveLength(2);
    expect(entries[0].source_commit).toHaveLength(40);
  });

  it('rejects malformed digests and unapproved source URLs', () => {
    expect(() => parseEvidenceCatalog({ ...fixture(), catalog_sha256: 'unknown' })).toThrow();
    expect(() =>
      parseEvidenceCatalog({
        ...fixture(),
        entries: [{ ...model('1'), source_path: '../../secrets' }, model('2')],
      }),
    ).toThrow();
    expect(() =>
      parseEvidenceCatalog({
        ...fixture(),
        entries: [
          { ...model('1'), workflow_url: 'https://attacker.invalid/actions/runs/123' },
          model('2'),
        ],
      }),
    ).toThrow();
  });

  it('rejects duplicate IDs, inflated counts and source-fingerprint drift', () => {
    expect(() =>
      parseEvidenceCatalog({ ...fixture(), entries: [model('1'), model('1')] }),
    ).toThrow();
    expect(() =>
      parseEvidenceCatalog({
        ...fixture(),
        counts: { ...fixture().counts, studies: 400 },
      }),
    ).toThrow();
    expect(() =>
      parseEvidenceCatalog({
        ...fixture(),
        entries: [{ ...model('1'), artifact_sha256: h('9') }, model('2')],
      }),
    ).toThrow();
  });

  it('filters without treating an unrelated operation as compatible', () => {
    const entries = parseEvidenceCatalog(fixture()).entries;
    expect(matchingEvidence(entries, { text: 'residual', kind: 'all' })).toHaveLength(1);
    expect(matchingEvidence(entries, { text: '', kind: 'rewrite_evaluation' })).toHaveLength(1);
    expect(matchingEvidence(entries, { operation: 'conv2d' })).toHaveLength(0);
    expect(matchingEvidence(entries, { operation: 'relu' })).toHaveLength(2);
  });

  it('shows matching binary lineage without equating semantic contracts', () => {
    const entries = parseEvidenceCatalog(fixture()).entries;
    expect(exactLineage(entries, entries[0])).toEqual([entries[1]]);
    expect(exactLineage(entries, { ...entries[0], native_binary_sha256: null })).toEqual([]);
    expect(
      exactLineage(entries, {
        ...entries[0],
        native_binary_sha256: h('5'),
      }),
    ).toEqual([]);
  });
});
