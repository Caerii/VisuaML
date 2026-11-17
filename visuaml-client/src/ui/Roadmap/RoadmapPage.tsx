import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  countRoadmapGates,
  displayProgress,
  parseRoadmap,
  portfolioProgress,
  workstreamProgress,
  type GateStatus,
  type RoadmapManifest,
  type RoadmapWorkstream,
} from './roadmapModel';
import './RoadmapPage.css';

const stages: { value: 'all' | GateStatus; label: string }[] = [
  { value: 'all', label: 'All workstreams' },
  { value: 'verified', label: 'Established' },
  { value: 'in-progress', label: 'In development' },
  { value: 'planned', label: 'Unfinished' },
];
const statusLabels: Record<GateStatus, string> = {
  verified: 'Verified gate',
  'in-progress': 'In development',
  planned: 'Not yet verified',
};
const sourceUrl = 'https://github.com/Caerii/VisuaML/blob/main/visuaml-client/public/roadmap.json';

function EvidenceLink({ label, url }: { label: string; url: string }) {
  return (
    <a href={url} target="_blank" rel="noreferrer" className="research-evidence-link">
      {label}
      <span aria-hidden="true"> ↗</span>
    </a>
  );
}

function ProgressBar({ value, label }: { value: number; label: string }) {
  return (
    <div
      className="research-progress-track"
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
    >
      <div className="research-progress-fill" style={{ width: String(value) + '%' }} />
    </div>
  );
}

function WorkstreamDetail({ stream }: { stream: RoadmapWorkstream }) {
  const percentage = workstreamProgress(stream);
  return (
    <article className="research-detail" aria-labelledby={'detail-' + stream.id}>
      <div className="research-detail-heading">
        <div>
          <p className="research-eyebrow">{stream.kicker}</p>
          <h2 id={'detail-' + stream.id}>{stream.title}</h2>
        </div>
        <div
          className="research-detail-number"
          aria-label={String(percentage) + '% of acceptance gates verified'}
        >
          {percentage}
          <span>%</span>
        </div>
      </div>
      <ProgressBar value={percentage} label={stream.title + ': ' + percentage + '% verified'} />
      <p className="research-lede">{stream.summary}</p>
      <div className="research-explanations">
        <div>
          <h3>Why it matters</h3>
          <p>{stream.impact}</p>
        </div>
        <div>
          <h3>What is established</h3>
          <p>{stream.now}</p>
        </div>
        <div>
          <h3>What remains</h3>
          <p>{stream.next}</p>
        </div>
      </div>
      <div className="research-gate-header">
        <h3>Acceptance ledger</h3>
        <span>Each criterion earns its weight only after verification</span>
      </div>
      <ol className="research-gates">
        {stream.criteria.map((gate) => (
          <li className="research-gate" key={gate.id}>
            <div className="research-gate-top">
              <span className={'research-status research-status--' + gate.status}>
                <span className="research-status-dot" aria-hidden="true" />
                {statusLabels[gate.status]}
              </span>
              <span className="research-gate-weight">{gate.weight} points</span>
            </div>
            <h4>{gate.title}</h4>
            <p>{gate.why}</p>
            {gate.evidence.length > 0 && (
              <div className="research-evidence">
                {gate.evidence.map((item) => (
                  <EvidenceLink key={item.url} label={item.label} url={item.url} />
                ))}
              </div>
            )}
          </li>
        ))}
      </ol>
    </article>
  );
}

/** Public by design: no auth, no mutation endpoint, no Python service dependency. */
export function RoadmapPage() {
  const [manifest, setManifest] = useState<RoadmapManifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [stage, setStage] = useState<'all' | GateStatus>('all');
  const [search, setSearch] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    fetch('/roadmap.json', { signal: controller.signal, cache: 'no-cache' })
      .then(async (response) => {
        if (!response.ok) throw new Error('The published roadmap could not be loaded.');
        return parseRoadmap((await response.json()) as unknown);
      })
      .then((report) => {
        if (controller.signal.aborted) return;
        setManifest(report);
        setSelected(report.workstreams[0]?.id ?? null);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(cause instanceof Error ? cause.message : 'Roadmap unavailable.');
      });
    return () => controller.abort();
  }, []);

  const filtered = useMemo(() => {
    if (!manifest) return [];
    const needle = search.trim().toLowerCase();
    return manifest.workstreams.filter((stream) => {
      const inStage = stage === 'all' || stream.criteria.some((gate) => gate.status === stage);
      const inSearch =
        !needle ||
        [
          stream.title,
          stream.summary,
          stream.kicker,
          stream.impact,
          stream.next,
          ...stream.criteria.flatMap((gate) => [gate.title, gate.why]),
        ].some((value) => value.toLowerCase().includes(needle));
      return inStage && inSearch;
    });
  }, [manifest, stage, search]);

  const active = filtered.find((stream) => stream.id === selected) ?? filtered[0];
  const count = manifest ? countRoadmapGates(manifest) : null;
  const overall = manifest ? displayProgress(portfolioProgress(manifest)) : null;

  return (
    <div className="research-roadmap">
      <header className="research-nav">
        <Link className="research-wordmark" to="/" aria-label="VisuaML home">
          <span className="research-monogram">
            V<span aria-hidden="true">/</span>
          </span>
          <span>VISUAML</span>
        </Link>
        <nav aria-label="Research site navigation">
          <Link to="/app">Workspace</Link>
          <Link to="/compiler">Compiler Lab</Link>
          <Link to="/architectures">Atlas</Link>
          <Link to="/docs">Documentation</Link>
          <Link to="/roadmap" aria-current="page" className="research-active">
            Roadmap
          </Link>
        </nav>
        <a className="research-nav-source" href={sourceUrl} target="_blank" rel="noreferrer">
          View source <span aria-hidden="true">↗</span>
        </a>
      </header>
      <main>
        <section className="research-hero" aria-labelledby="research-title">
          <div className="research-hero-kicker">
            <span className="research-live-mark" aria-hidden="true" />
            OPEN RESEARCH · A VERSIONED PUBLIC RECORD
          </div>
          <p className="research-issue">01 / THE ACCOUNTABILITY OF COMPUTATION</p>
          <h1 id="research-title">
            A roadmap measured
            <br />
            in <em>evidence.</em>
          </h1>
          <p className="research-intro">
            A computational idea should not be imprisoned within its first implementation. Our work
            is to make its structure intelligible, its transformations testable, and its execution
            portable. This is an open account of how far we have come—and the considerable work
            still ahead.
          </p>
          <div className="research-hero-meta">
            <span>RESEARCH MILESTONE V27</span>
            <span aria-hidden="true">/</span>
            <span>OPEN SOURCE · MIT</span>
            <span aria-hidden="true">/</span>
            <span>VERSIONED IN GIT</span>
          </div>
        </section>
        {error && (
          <div className="research-alert" role="alert">
            <h2>Roadmap temporarily unavailable</h2>
            <p>{error} The last reviewed data can still be inspected in the public repository.</p>
            <a href={sourceUrl} target="_blank" rel="noreferrer">
              Open the source ledger ↗
            </a>
          </div>
        )}
        {!manifest && !error && (
          <div className="research-loading" role="status">
            Loading the published evidence ledger…
          </div>
        )}
        {manifest && count && (
          <>
            <section className="research-overview" aria-label="Roadmap overview">
              <div className="research-overall-score">
                <span className="research-stat-label">WEIGHTED ACCEPTANCE</span>
                <div className="research-overall-value">
                  {overall}
                  <span>%</span>
                </div>
                <ProgressBar
                  value={overall ?? 0}
                  label={String(overall) + '% of published weighted acceptance gates verified'}
                />
                <p>
                  This measures completed, evidence-backed gates under our published rubric—not a
                  prediction of when all research will finish.
                </p>
              </div>
              <div className="research-overview-metrics">
                <div>
                  <strong>{String(manifest.workstreams.length).padStart(2, '0')}</strong>
                  <span>RESEARCH AREAS</span>
                </div>
                <div>
                  <strong>{count.verified}</strong>
                  <span>VERIFIED CRITERIA</span>
                </div>
                <div>
                  <strong>{count.inProgress + count.planned}</strong>
                  <span>UNVERIFIED CRITERIA</span>
                </div>
              </div>
              <div className="research-provenance">
                <span className="research-stat-label">LAST RESEARCH REVIEW</span>
                <strong>
                  {new Date(manifest.asOf + 'T12:00:00Z').toLocaleDateString('en-US', {
                    timeZone: 'UTC',
                    day: 'numeric',
                    month: 'long',
                    year: 'numeric',
                  })}
                </strong>
                <p>Source baseline</p>
                <a
                  href={'https://github.com/Caerii/VisuaML/commit/' + manifest.referenceCommit}
                  target="_blank"
                  rel="noreferrer"
                >
                  {manifest.referenceCommit.slice(0, 12)} ↗
                </a>
              </div>
            </section>
            <section className="research-ledger-section" aria-labelledby="ledger-title">
              <div className="research-section-intro">
                <div>
                  <p className="research-eyebrow">THE WORK, NOT THE MYTH</p>
                  <h2 id="ledger-title">Eight connected frontiers.</h2>
                </div>
                <p>
                  Select a field to see every acceptance gate, how its percentage is earned, which
                  results can be inspected, and what has not been demonstrated.
                </p>
              </div>
              <div className="research-filters">
                <label>
                  <span>FIND A FIELD OR CRITERION</span>
                  <input
                    type="search"
                    placeholder="Search the research ledger"
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                  />
                </label>
                <label>
                  <span>SHOW</span>
                  <select
                    value={stage}
                    onChange={(event) => setStage(event.target.value as typeof stage)}
                  >
                    {stages.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <span className="research-filter-count">
                  {filtered.length} of {manifest.workstreams.length} fields
                </span>
              </div>
              {filtered.length > 0 ? (
                <div className="research-ledger-layout">
                  <div
                    className="research-workstream-list"
                    role="group"
                    aria-label="Research workstreams"
                  >
                    {filtered.map((stream, index) => {
                      const progress = workstreamProgress(stream);
                      const isActive = stream.id === active?.id;
                      return (
                        <button
                          key={stream.id}
                          type="button"
                          className={'research-workstream' + (isActive ? ' is-active' : '')}
                          onClick={() => setSelected(stream.id)}
                          aria-pressed={isActive}
                        >
                          <span className="research-workstream-index">
                            {String(index + 1).padStart(2, '0')}
                          </span>
                          <span className="research-workstream-content">
                            <span className="research-workstream-kicker">{stream.kicker}</span>
                            <strong>{stream.title}</strong>
                            <span className="research-workstream-weight">
                              {stream.weight}% portfolio weight
                            </span>
                            <ProgressBar
                              value={progress}
                              label={stream.title + ': ' + progress + '% complete'}
                            />
                          </span>
                          <span className="research-workstream-score">
                            {progress}
                            <span>%</span>
                          </span>
                        </button>
                      );
                    })}
                  </div>
                  {active && <WorkstreamDetail key={active.id} stream={active} />}
                </div>
              ) : (
                <div className="research-no-results">
                  <p>Nothing matches those filters. The research is still here.</p>
                  <button
                    onClick={() => {
                      setStage('all');
                      setSearch('');
                    }}
                  >
                    Show all fields
                  </button>
                </div>
              )}
            </section>
            <section className="research-method" aria-labelledby="method-title">
              <div className="research-method-heading">
                <span className="research-eyebrow">READING THE NUMBERS</span>
                <h2 id="method-title">
                  A percentage is
                  <br />
                  <em>an argument.</em>
                </h2>
              </div>
              <div className="research-method-content">
                <h3>{manifest.methodology.name}</h3>
                <p>{manifest.methodology.text}</p>
                <p>{manifest.methodology.caveat}</p>
                <div className="research-formula">
                  <span>THE ACCOUNTING RULE</span>
                  <code>Σ (workstream weight × verified gate points) ÷ 100</code>
                </div>
                <h3>How often is this updated?</h3>
                <p>{manifest.methodology.refresh}</p>
                <div className="research-method-links">
                  <EvidenceLink label="Inspect the canonical JSON" url={sourceUrl} />
                  <EvidenceLink
                    label="Review the verification methodology"
                    url="https://github.com/Caerii/VisuaML/blob/main/docs/VERIFICATION.md"
                  />
                  <EvidenceLink
                    label="Inspect compiler source"
                    url="https://github.com/Caerii/VisuaML/tree/main/visuaml-client/backend/visuaml/compiler_lab"
                  />
                </div>
              </div>
            </section>
          </>
        )}
      </main>
      <footer className="research-footer">
        <div>
          <Link to="/">VISUAML</Link>
          <span>Computation, made legible.</span>
        </div>
        <p>
          An open-source research instrument in progress. No speculative outcome is represented as a
          verified result.
        </p>
        <a href="https://github.com/Caerii/VisuaML" target="_blank" rel="noreferrer">
          Source ↗
        </a>
      </footer>
    </div>
  );
}
