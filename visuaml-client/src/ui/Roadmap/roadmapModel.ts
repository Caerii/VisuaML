/** Pure, validated public roadmap arithmetic. Percentages are not estimates. */
export type GateStatus = 'verified' | 'in-progress' | 'planned';
export type RoadmapEvidence = { label: string; url: string };
export type RoadmapGate = {
  id: string; title: string; weight: number;
  status: GateStatus; why: string; evidence: RoadmapEvidence[];
};
export type RoadmapWorkstream = {
  id: string; title: string; kicker: string; weight: number;
  summary: string; impact: string; now: string; next: string;
  criteria: RoadmapGate[];
};
export type RoadmapManifest = {
  schema: 'visuaml.public-research-roadmap.v1';
  asOf: string; referenceCommit: string; project: string;
  title: string; subtitle: string;
  methodology: { name: string; text: string; caveat: string; refresh: string };
  workstreams: RoadmapWorkstream[];
};
const object = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const text = (value: unknown): value is string =>
  typeof value === 'string' && value.trim().length > 0;
const percentage = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value > 0 && value <= 100;
const isGateStatus = (value: unknown): value is GateStatus =>
  value === 'verified' || value === 'in-progress' || value === 'planned';

/** Only verified criteria contribute to progress, not unverified intentions. */
export function workstreamProgress(stream: RoadmapWorkstream): number {
  return stream.criteria.reduce(
    (sum, gate) => sum + (gate.status === 'verified' ? gate.weight : 0), 0,
  );
}
export function portfolioProgress(manifest: RoadmapManifest): number {
  return manifest.workstreams.reduce(
    (sum, stream) => sum + stream.weight * workstreamProgress(stream), 0,
  ) / 100;
}
export function countRoadmapGates(manifest: RoadmapManifest) {
  const criteria = manifest.workstreams.flatMap((stream) => stream.criteria);
  return {
    all: criteria.length,
    verified: criteria.filter((gate) => gate.status === 'verified').length,
    inProgress: criteria.filter((gate) => gate.status === 'in-progress').length,
    planned: criteria.filter((gate) => gate.status === 'planned').length,
  };
}

export function parseRoadmap(value: unknown): RoadmapManifest {
  if (
    !object(value) ||
    value.schema !== 'visuaml.public-research-roadmap.v1' ||
    !text(value.project) || !text(value.title) || !text(value.subtitle) ||
    !text(value.asOf) || !/^\d{4}-\d{2}-\d{2}$/.test(value.asOf) ||
    !text(value.referenceCommit) || !/^[a-f0-9]{40}$/.test(value.referenceCommit) ||
    !object(value.methodology) ||
    !text(value.methodology.name) || !text(value.methodology.text) ||
    !text(value.methodology.caveat) || !text(value.methodology.refresh) ||
    !Array.isArray(value.workstreams) || value.workstreams.length === 0 ||
    value.workstreams.length > 20
  ) {
    throw new Error('Invalid roadmap manifest or provenance');
  }
  const ids = new Set<string>();
  let total = 0;
  for (const stream of value.workstreams) {
    if (
      !object(stream) || !text(stream.id) ||
      !/^[a-z][a-z0-9-]*$/.test(stream.id) || ids.has(stream.id) ||
      !text(stream.title) || !text(stream.kicker) || !text(stream.summary) ||
      !text(stream.impact) || !text(stream.now) || !text(stream.next) ||
      !percentage(stream.weight) || !Array.isArray(stream.criteria) ||
      stream.criteria.length < 2 || stream.criteria.length > 25
    ) {
      throw new Error('Invalid or duplicate workstream');
    }
    ids.add(stream.id);
    total += stream.weight;
    let gates = 0;
    const gateIds = new Set<string>();
    for (const gate of stream.criteria) {
      if (
        !object(gate) || !text(gate.id) ||
        !/^[a-z][a-z0-9-]*$/.test(gate.id) || gateIds.has(gate.id) ||
        !text(gate.title) || !text(gate.why) || !isGateStatus(gate.status) ||
        !percentage(gate.weight) || !Array.isArray(gate.evidence) ||
        !gate.evidence.every(
          (entry: unknown) =>
            object(entry) && text(entry.label) && text(entry.url) &&
            /^https:\/\/github\.com\/Caerii\/VisuaML\//.test(entry.url),
        ) ||
        (gate.status === 'verified' && gate.evidence.length === 0)
      ) {
        throw new Error('Invalid acceptance gate or evidence');
      }
      gateIds.add(gate.id);
      gates += gate.weight;
    }
    if (gates !== 100) throw new Error('Each workstream must allocate exactly 100 points');
  }
  if (total !== 100) throw new Error('Portfolio weights must sum to 100');
  return value as RoadmapManifest;
}
export const displayProgress = (score: number) => Math.round(score);
