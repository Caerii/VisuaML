import { describe, expect, it } from 'vitest';
import {
  countRoadmapGates,
  parseRoadmap,
  portfolioProgress,
  workstreamProgress,
} from './roadmapModel';

const fixture = {
  schema: 'visuaml.public-research-roadmap.v1',
  asOf: '2026-11-16',
  referenceCommit: '3ed102286817a0bff76121fc13fa907dd86e762a',
  project: 'VisuaML',
  title: 'A testable roadmap',
  subtitle: 'Progress is evidence.',
  methodology: {
    name: 'Gates',
    text: 'Only evidence counts',
    caveat: 'A snapshot',
    refresh: 'On deployments',
  },
  workstreams: [
    {
      id: 'first',
      title: 'First',
      kicker: 'MATH',
      weight: 60,
      summary: 'One',
      impact: 'Because',
      now: 'Done',
      next: 'More',
      criteria: [
        {
          id: 'verified',
          title: 'Certified',
          weight: 40,
          status: 'verified',
          why: 'Has evidence',
          evidence: [
            {
              label: 'Commit',
              url: 'https://github.com/Caerii/VisuaML/commit/abc',
            },
          ],
        },
        {
          id: 'future',
          title: 'Research',
          weight: 60,
          status: 'in-progress',
          why: 'Not proven',
          evidence: [],
        },
      ],
    },
    {
      id: 'second',
      title: 'Second',
      kicker: 'EXPERIMENT',
      weight: 40,
      summary: 'Two',
      impact: 'Because',
      now: 'Done',
      next: 'More',
      criteria: [
        {
          id: 'yes',
          title: 'Checked',
          weight: 70,
          status: 'verified',
          why: 'Proof',
          evidence: [
            {
              label: 'CI',
              url: 'https://github.com/Caerii/VisuaML/actions',
            },
          ],
        },
        {
          id: 'no',
          title: 'Later',
          weight: 30,
          status: 'planned',
          why: 'Future',
          evidence: [],
        },
      ],
    },
  ],
};

describe('public research roadmap', () => {
  it('awards credit only to evidence-backed verified gates', () => {
    const manifest = parseRoadmap(fixture);
    expect(workstreamProgress(manifest.workstreams[0])).toBe(40);
    expect(workstreamProgress(manifest.workstreams[1])).toBe(70);
    expect(portfolioProgress(manifest)).toBe(52);
    expect(countRoadmapGates(manifest)).toEqual({
      all: 4,
      verified: 2,
      inProgress: 1,
      planned: 1,
    });
  });
  it('rejects verified gates without evidence and mismatched portfolio weights', () => {
    const missing = structuredClone(fixture);
    missing.workstreams[0].criteria[0].evidence = [];
    expect(() => parseRoadmap(missing)).toThrow();
    const badWeights = structuredClone(fixture);
    badWeights.workstreams[0].weight = 20;
    expect(() => parseRoadmap(badWeights)).toThrow();
  });
  it('rejects untrusted URLs, forged source hashes and duplicate identifiers', () => {
    const untrusted = structuredClone(fixture);
    untrusted.workstreams[0].criteria[0].evidence[0].url = 'javascript:alert(1)';
    expect(() => parseRoadmap(untrusted)).toThrow();
    const wrongHash = structuredClone(fixture);
    wrongHash.referenceCommit = 'unverified';
    expect(() => parseRoadmap(wrongHash)).toThrow();
    const duplicate = structuredClone(fixture);
    duplicate.workstreams[1].id = 'first';
    expect(() => parseRoadmap(duplicate)).toThrow();
  });
});
