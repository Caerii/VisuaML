import { describe, expect, it } from 'vitest';
import { parseEvidenceSnapshot } from './evidenceModel';

const sha = (character: string) => character.repeat(64);

const example = () => ({
  schema: 'visuaml.public-evidence-snapshot.v1',
  asOf: '2026-11-17',
  sourceCommit: 'a'.repeat(40),
  workflowRun: 'https://github.com/Caerii/VisuaML/actions/runs/12345',
  methodology: 'Finite independent comparison',
  evidenceClass: 'Not a formal proof',
  models: [
    {
      family: 'MLP',
      kind: 'Synthetic export',
      operations: ['matmul', 'relu'],
      maximumAbsoluteError: 0,
      secondInputMaximumAbsoluteError: 1e-7,
      semanticSha256: sha('b'),
      generatedCppSha256: sha('c'),
      binarySha256: sha('d'),
      compileMs: 100,
      warmMedianMs: 0.02,
    },
  ],
  transformation: {
    rule: 'relu-idempotence-v1',
    numericalStatus: 'passed_finite_witnesses',
    evidenceSha256: sha('e'),
    experimentalStatus: 'exploratory_timing_after_finite_numerical_witnesses',
    medianRatioBeforeOverAfter: 0.962,
    interpretation: 'Slightly slower in this sample',
  },
  limitations: ['One runner', 'Not signed'],
});

describe('versioned public evidence snapshot', () => {
  it('accepts the exact bounded published layout', () => {
    const input = example();
    const accepted = parseEvidenceSnapshot(input);
    expect(accepted.models[0].semanticSha256).toBe(sha('b'));
    expect(accepted.sourceCommit).toHaveLength(40);
  });

  it('rejects unknown schema and malformed hashes', () => {
    expect(() => parseEvidenceSnapshot({ ...example(), schema: 'other' })).toThrow();
    expect(() => parseEvidenceSnapshot({ ...example(), sourceCommit: sha('a') })).toThrow();
    expect(() =>
      parseEvidenceSnapshot({
        ...example(),
        models: [{ ...example().models[0], binarySha256: 'not-a-hash' }],
      }),
    ).toThrow();
  });

  it('rejects untrusted CI links and invalid numerical evidence', () => {
    expect(() =>
      parseEvidenceSnapshot({ ...example(), workflowRun: 'https://attacker.example/123' }),
    ).toThrow();
    expect(() =>
      parseEvidenceSnapshot({
        ...example(),
        models: [{ ...example().models[0], maximumAbsoluteError: Number.NaN }],
      }),
    ).toThrow();
    expect(() =>
      parseEvidenceSnapshot({
        ...example(),
        transformation: { ...example().transformation, medianRatioBeforeOverAfter: -1 },
      }),
    ).toThrow();
  });
});
