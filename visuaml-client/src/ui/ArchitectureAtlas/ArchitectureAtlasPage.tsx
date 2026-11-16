/** Versioned, operator-level planning atlas. Never labels model families executable. */
import { useEffect, useMemo, useState } from 'react';
import { FamilyBlockEvidence } from './FamilyBlockEvidence';
import { DecoderAOTEvidence } from './DecoderAOTEvidence';
import { StackedDecoderEvidence } from './StackedDecoderEvidence';
import { SemanticDecoderEvidence } from './SemanticDecoderEvidence';
import './ArchitectureAtlas.css';

type Coverage = {
  architecture_id: string;
  status: 'candidate-for-validation' | 'blocked-by-contract';
  covered_operators: string[];
  missing_operators: string[];
  missing_features: string[];
  coverage_fraction: number;
};
type Family = {
  id: string;
  name: string;
  domain: string;
  priority: 1 | 2 | 3;
  operators: string[];
  features: string[];
  exemplars: string[];
  note: string;
  coverage: Coverage;
};
type Gap = {
  capability: string;
  weighted_incidence: number;
  families: string[];
  metric: string;
};
type Atlas = {
  schema: string;
  claim: string;
  available_ops: string[];
  available_features: string[];
  families: Family[];
  priority_gaps: Gap[];
  sha256: string;
};

function validateAtlas(data: unknown): Atlas {
  if (!data || typeof data !== 'object') throw new Error('Atlas response is not an object');
  const atlas = data as Partial<Atlas>;
  if (
    atlas.schema !== 'visuaml.architecture-atlas.v1' ||
    !Array.isArray(atlas.families) ||
    !Array.isArray(atlas.priority_gaps) ||
    !Array.isArray(atlas.available_ops) ||
    typeof atlas.sha256 !== 'string' ||
    atlas.families.some((f) => !f.id || !f.coverage || !Array.isArray(f.coverage.missing_operators))
  ) {
    throw new Error('Invalid or outdated architecture atlas contract');
  }
  return atlas as Atlas;
}

const priorityName = (priority: number) => {
  if (priority === 1) return 'Core';
  if (priority === 2) return 'Research';
  return 'Exploratory';
};

export function ArchitectureAtlasPage() {
  const [atlas, setAtlas] = useState<Atlas | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [domain, setDomain] = useState('all');
  const [selectedId, setSelectedId] = useState('decoder_transformer');
  const [showMissingOnly, setShowMissingOnly] = useState(false);

  useEffect(() => {
    let active = true;
    fetch('/architecture_atlas.v1.json')
      .then(async (response) => {
        if (!response.ok) throw new Error(`Atlas request returned ${response.status}`);
        return validateAtlas(await response.json());
      })
      .then((data) => {
        if (active) setAtlas(data);
      })
      .catch((error: unknown) => {
        if (active) setFailure(error instanceof Error ? error.message : 'Unknown atlas error');
      });
    return () => {
      active = false;
    };
  }, []);

  const domains = useMemo(
    () => [...new Set(atlas?.families.map((f) => f.domain) ?? [])].sort(),
    [atlas],
  );
  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    return (atlas?.families ?? [])
      .filter((f) => domain === 'all' || f.domain === domain)
      .filter((f) => !showMissingOnly || f.coverage.status === 'blocked-by-contract')
      .filter(
        (f) =>
          !query ||
          `${f.id} ${f.name} ${f.domain} ${f.exemplars.join(' ')} ${f.operators.join(' ')}`
            .toLowerCase()
            .includes(query),
      )
      .sort(
        (a, b) =>
          a.priority - b.priority ||
          b.coverage.coverage_fraction - a.coverage.coverage_fraction ||
          a.name.localeCompare(b.name),
      );
  }, [atlas, search, domain, showMissingOnly]);
  const selected = (atlas?.families ?? []).find((f) => f.id === selectedId) ?? filtered[0];
  const primitiveCandidates = (atlas?.families ?? []).filter(
    (f) => f.coverage.status === 'candidate-for-validation',
  ).length;

  return (
    <div className="atlas-root">
      <header className="atlas-header">
        <div>
          <div className="atlas-nav">
            <a href="/compiler">← Compiler Lab</a>
            <a href="/app">VisuaML canvas</a>
          </div>
          <p className="atlas-eyebrow">VISUAML RESEARCH · CAPABILITY OBSERVATORY</p>
          <h1>Neural Architecture Atlas</h1>
          <p className="atlas-subtitle">
            Know precisely which computational primitives we can represent, which semantics are
            missing, and what to build next.
          </p>
        </div>
        <div className="atlas-metadata">
          <strong>Version 11 / operator contracts</strong>
          <span>Inspection readiness ≠ full-model compilation</span>
        </div>
      </header>
      <div className="atlas-warning" role="note">
        This is a <strong>planning matrix</strong>, not a model leaderboard or compatibility
        certification. Even “candidate for validation” means only that declared primitive names are
        present. Real checkpoints still need capture, guards, execution, and differential tests. No
        Ayo hardware is connected.
      </div>
      {failure && (
        <p role="alert" className="atlas-error">
          {failure}
        </p>
      )}
      {!atlas ? (
        <p className="atlas-loading">
          {failure
            ? 'Could not load the versioned atlas.'
            : 'Loading local architecture coverage snapshot…'}
        </p>
      ) : (
        <>
          <section className="atlas-stat-grid" aria-label="Coverage overview">
            <div>
              <small>Architecture families</small>
              <strong>{atlas.families.length}</strong>
              <span>Catalogued motifs</span>
            </div>
            <div>
              <small>Domains</small>
              <strong>{domains.length}</strong>
              <span>Language, vision, science and beyond</span>
            </div>
            <div>
              <small>Primitive-only candidates</small>
              <strong>{primitiveCandidates}</strong>
              <span>Not full-model certification</span>
            </div>
            <div>
              <small>NumPy-AOT operators</small>
              <strong>{atlas.available_ops.length}</strong>
              <span>Declared interpreter names</span>
            </div>
          </section>
          <section className="atlas-workspace">
            <div className="atlas-list">
              <div className="atlas-list-heading">
                <h2>Architecture families</h2>
                <span>{filtered.length} shown</span>
              </div>
              <label className="atlas-search-label" htmlFor="architecture-search">
                Search architecture or operator
              </label>
              <input
                id="architecture-search"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="MoE, diffusion, Mamba, spiking…"
              />
              <div className="atlas-controls">
                <select
                  aria-label="Filter by architecture domain"
                  value={domain}
                  onChange={(e) => setDomain(e.target.value)}
                >
                  <option value="all">All domains</option>
                  {domains.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
                <label className="atlas-toggle">
                  <input
                    type="checkbox"
                    checked={showMissingOnly}
                    onChange={(e) => setShowMissingOnly(e.target.checked)}
                  />{' '}
                  Has gaps
                </label>
              </div>
              <div className="atlas-list-scroll">
                {filtered.map((f) => (
                  <button
                    type="button"
                    key={f.id}
                    className={`atlas-family ${selected?.id === f.id ? 'selected' : ''}`}
                    onClick={() => setSelectedId(f.id)}
                    aria-pressed={selected?.id === f.id}
                  >
                    <span className="atlas-family-top">
                      <strong>{f.name}</strong>
                      <small>{priorityName(f.priority)}</small>
                    </span>
                    <span className="atlas-family-bottom">
                      <small>{f.domain}</small>
                      <small>
                        {f.coverage.covered_operators.length}/{f.operators.length} named ops
                      </small>
                    </span>
                    <span className="atlas-progress">
                      <span style={{ width: `${f.coverage.coverage_fraction * 100}%` }} />
                    </span>
                  </button>
                ))}
                {filtered.length === 0 && (
                  <p className="atlas-empty">No architecture matches these filters.</p>
                )}
              </div>
            </div>
            <div className="atlas-details">
              {selected ? (
                <>
                  <div className="atlas-detail-meta">
                    <span>{selected.domain}</span>
                    <span>Priority {selected.priority}</span>
                    <span>{selected.coverage.status.replace(/-/g, ' ')}</span>
                  </div>
                  <h2>{selected.name}</h2>
                  <p className="atlas-note">{selected.note}</p>
                  <div className="atlas-primitives">
                    <h3>Declared computational primitives</h3>
                    <p className="atlas-muted">
                      Green means that an operator spelling exists in the portable runtime; it does
                      not prove this architecture has executed.
                    </p>
                    <div className="atlas-tags">
                      {selected.operators.map((op) => (
                        <span
                          key={op}
                          className={
                            selected.coverage.covered_operators.includes(op)
                              ? 'available'
                              : 'missing'
                          }
                        >
                          {selected.coverage.covered_operators.includes(op) ? '✓' : '×'} {op}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="atlas-primitives">
                    <h3>Additional semantic contracts</h3>
                    {selected.features.length ? (
                      <div className="atlas-tags">
                        {selected.features.map((feat) => (
                          <span
                            key={feat}
                            className={
                              selected.coverage.missing_features.includes(feat)
                                ? 'missing'
                                : 'available'
                            }
                          >
                            {feat.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <p className="atlas-muted">
                        No additional feature requirement is declared for this representative motif.
                      </p>
                    )}
                  </div>
                  <div className="atlas-primitives">
                    <h3>Representative examples</h3>
                    <p>{selected.exemplars.join(' · ')}</p>
                  </div>
                  <FamilyBlockEvidence architectureId={selected.id} />
                  <DecoderAOTEvidence architectureId={selected.id} />
                  <StackedDecoderEvidence architectureId={selected.id} />
                  <SemanticDecoderEvidence architectureId={selected.id} />
                  <div className="atlas-acceptance">
                    <h3>What would establish genuine support?</h3>
                    <p>
                      A real model checkpoint, safe input fixture, successful capture, operator
                      legalization, portable or hardware-backed execution, shape and dtype guards,
                      and numerical/functional comparison against the reference implementation.
                    </p>
                  </div>
                </>
              ) : (
                <p>Select an architecture to inspect its contracts.</p>
              )}
            </div>
            <aside className="atlas-agenda">
              <h2>Next capability investments</h2>
              <p className="atlas-muted">
                Weighted incidence across catalogued workloads—not performance gain, popularity
                measurement, or implementation estimate.
              </p>
              {atlas.priority_gaps.slice(0, 12).map((gap, index) => (
                <div key={gap.capability} className="atlas-gap">
                  <span className="atlas-gap-rank">{index + 1}</span>
                  <div>
                    <strong>{gap.capability.replace(':', ' · ').replace(/_/g, ' ')}</strong>
                    <small>
                      {gap.families.length} listed families · incidence {gap.weighted_incidence}
                    </small>
                  </div>
                </div>
              ))}
              <div className="atlas-evidence">
                <h3>Evidence ladder</h3>
                <p>
                  Catalogued → Operator audit → Captured microfixture → Compiled artifact →
                  Checkpoint parity → Device measurement.
                </p>
              </div>
            </aside>
          </section>
          <footer className="atlas-footer">
            Snapshot SHA-256: <code>{atlas.sha256}</code>. Generated from the v11 operator registry;
            model-family coverage must be validated separately.
          </footer>
        </>
      )}
    </div>
  );
}
