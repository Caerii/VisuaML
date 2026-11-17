import { useEffect, useMemo, useState } from 'react';
import {
  exactLineage,
  matchingEvidence,
  parseEvidenceCatalog,
  type EvidenceCatalog,
  type EvidenceEntry,
} from './libraryModel';
import './ResearchEvidencePage.css';
import './ResearchLibraryPage.css';

const descriptions: Record<string, string> = {
  attention_context_sweep: 'Attention / context experiment',
  native_generated_graph: 'Graph-generated CPU witness',
  native_exported_graph: 'Exported native program',
  rewrite_evaluation: 'Transformation experiment',
  optimization_measurement: 'Experimental optimization measurement',
};
const describe = (kind: string): string =>
  descriptions[kind] || kind.replace(/_/g, ' ');

const brief = (sha: string | null) => (sha ? sha.slice(0, 12) : 'Not recorded');

export function ResearchLibraryPage() {
  const [catalog, setCatalog] = useState<EvidenceCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [phrase, setPhrase] = useState('');
  const [kind, setKind] = useState('all');
  const [operation, setOperation] = useState('all');
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    const abort = new AbortController();
    const load = async () => {
      try {
        const response = await fetch('/evidence/library-v34.json', {
          signal: abort.signal,
          cache: 'no-store',
        });
        if (!response.ok) throw new Error('The published library is not available');
        const value: unknown = await response.json();
        const checked = parseEvidenceCatalog(value);
        if (!abort.signal.aborted) {
          setCatalog(checked);
          setSelectedId(checked.entries[0]?.entry_id ?? null);
        }
      } catch (cause) {
        if (!abort.signal.aborted) {
          setError(cause instanceof Error ? cause.message : 'Invalid research catalog');
        }
      } finally {
        if (!abort.signal.aborted) setLoading(false);
      }
    };
    void load();
    return () => abort.abort();
  }, []);

  const results = useMemo(
    () => matchingEvidence(catalog?.entries ?? [], { text: phrase, kind, operation }),
    [catalog, phrase, kind, operation],
  );
  const selected: EvidenceEntry | null = useMemo(
    () => results.find((entry) => entry.entry_id === selectedId) ?? results[0] ?? null,
    [results, selectedId],
  );
  const similar = useMemo(
    () => (selected ? exactLineage(catalog?.entries ?? [], selected) : []),
    [catalog, selected],
  );
  const operations = useMemo(
    () => Array.from(new Set(catalog?.entries.flatMap((entry) => entry.op_sequence) ?? [])).sort(),
    [catalog],
  );

  return (
    <main className="research-evidence research-library">
      <header className="research-evidence-header">
        <nav aria-label="Research navigation">
          <a href="/research">Research evidence</a>
          <a href="/compiler">Compiler Lab</a>
          <a href="/roadmap">Public roadmap</a>
        </nav>
        <p className="research-evidence-eyebrow">VISUAML · CUMULATIVE RESEARCH KNOWLEDGE</p>
        <h1>The computational evidence library.</h1>
        <p className="research-evidence-intro">
          Discover the exact source, mathematical operations, target assumptions, and limitations
          behind our compiler experiments. Shared hashes connect past implementations; they do not
          automatically prove correctness or authorize a performance claim.
        </p>
      </header>

      {loading && <p role="status">Validating the reviewed research catalog…</p>}
      {error && (
        <section className="research-evidence-unavailable" role="alert">
          <h2>Evidence library unavailable</h2>
          <p>{error}. The GitHub repository does not guarantee this site has deployed.</p>
        </section>
      )}

      {catalog && (
        <>
          <section className="research-library-overview" aria-label="Library coverage">
            <div>
              <span className="research-evidence-label">INDEXED OBSERVATIONS</span>
              <strong>{catalog.counts.studies}</strong>
            </div>
            <div>
              <span className="research-evidence-label">SOURCE ARTIFACTS</span>
              <strong>{catalog.counts.source_artifacts}</strong>
            </div>
            <div>
              <span className="research-evidence-label">CATALOG SHA-256</span>
              <code title={catalog.catalog_sha256}>{brief(catalog.catalog_sha256)}</code>
            </div>
          </section>

          <section className="research-evidence-section">
            <p className="research-evidence-eyebrow">I. SEARCH THE EXPERIMENTS</p>
            <h2>Find what has actually been measured.</h2>
            <div className="research-library-controls">
              <label>
                Search research
                <input
                  type="search"
                  placeholder="Graph, operator, model, evidence class…"
                  value={phrase}
                  onChange={(event) => setPhrase(event.target.value)}
                />
              </label>
              <label>
                Type of study
                <select value={kind} onChange={(event) => setKind(event.target.value)}>
                  <option value="all">All studies</option>
                  {Object.keys(catalog.counts.kinds)
                    .sort()
                    .map((value) => (
                      <option value={value} key={value}>
                        {describe(value)}
                      </option>
                    ))}
                </select>
              </label>
              <label>
                Operator
                <select value={operation} onChange={(event) => setOperation(event.target.value)}>
                  <option value="all">All operators</option>
                  {operations.map((value) => (
                    <option value={value} key={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p role="status" className="research-library-count">
              Showing {results.length} of {catalog.entries.length} historical observations
            </p>

            <div className="research-library-layout">
              <div className="research-library-list">
                {results.length === 0 && <p>No matching research observations.</p>}
                {results.map((entry) => (
                  <button
                    key={entry.entry_id}
                    type="button"
                    className={`research-library-item ${selectedId === entry.entry_id ? 'selected' : ''}`}
                    onClick={() => setSelectedId(entry.entry_id)}
                    aria-pressed={selectedId === entry.entry_id}
                  >
                    <span className="research-evidence-label">{describe(entry.kind)}</span>
                    <strong>{entry.label}</strong>
                    <span className="research-library-item-foot">
                      <span>{entry.evidence_class.replace(/_/g, ' ')}</span>
                      <code>{brief(entry.artifact_sha256)}</code>
                    </span>
                  </button>
                ))}
              </div>

              <aside className="research-library-detail" aria-label="Selected evidence">
                {selected ? (
                  <>
                    <span className="research-evidence-label">{describe(selected.kind)}</span>
                    <h3>{selected.label}</h3>
                    <p className="research-library-detail-status">
                      {selected.status.replace(/_/g, ' ')}
                    </p>
                    <h4>Admitted mathematical operations</h4>
                    <div className="research-evidence-ops">
                      {selected.op_sequence.map((op, i) => (
                        <span key={i}>
                          <code>{op}</code>
                        </span>
                      ))}
                    </div>
                    <h4>Context and assumptions</h4>
                    <dl className="research-library-key-values">
                      {Object.entries(selected.context).map(([key, value]) => (
                        <div key={key}>
                          <dt>{key.replace(/_/g, ' ')}</dt>
                          <dd>{value}</dd>
                        </div>
                      ))}
                    </dl>
                    <h4>Observed metrics</h4>
                    <dl className="research-library-key-values">
                      {Object.entries(selected.result).map(([key, value]) => (
                        <div key={key}>
                          <dt>{key.replace(/_/g, ' ')}</dt>
                          <dd>{Number(value).toPrecision(6)}</dd>
                        </div>
                      ))}
                    </dl>
                    <h4>Source and native identity</h4>
                    <dl className="research-library-key-values fingerprints">
                      {(
                        [
                          ['IR digest', selected.semantic_sha256],
                          ['Generated C++', selected.generated_cpp_sha256],
                          ['Native binary', selected.native_binary_sha256],
                          ['Pinned checkpoint', selected.checkpoint_sha256],
                        ] as const
                      ).map(([label, hash]) => (
                        <div key={label}>
                          <dt>{label}</dt>
                          <dd>
                            <code title={hash || ''}>{brief(hash)}</code>
                          </dd>
                        </div>
                      ))}
                    </dl>
                    <p className="research-library-links">
                      <a
                        href={`https://github.com/Caerii/VisuaML/blob/main/${selected.source_path}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Read archived observation
                      </a>
                      <a href={selected.workflow_url} target="_blank" rel="noreferrer">
                        Inspect original CI
                      </a>
                    </p>
                    <h4>Exact native binary lineage</h4>
                    {similar.length ? (
                      <div className="research-library-lineage">
                        {similar.map((entry) => (
                          <button
                            key={entry.entry_id}
                            onClick={() => setSelectedId(entry.entry_id)}
                          >
                            {entry.label} <span>same binary fingerprint</span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p>No other indexed study records this identical native binary hash.</p>
                    )}
                    <details>
                      <summary>Scientific limitations and interpretation</summary>
                      <ul>
                        {selected.limitations.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </details>
                  </>
                ) : (
                  <p>Select an experiment to inspect its admission and provenance.</p>
                )}
              </aside>
            </div>
          </section>
          <section className="research-evidence-section research-library-interpretation">
            <p className="research-evidence-eyebrow">II. KNOWLEDGE THAT COMPOUNDS</p>
            <h2>Identity makes evidence findable—not universally transferable.</h2>
            <p>
              Matching generated source or native binary fingerprints reveals shared code across
              independent experiments. Reusing a claim about mathematical behavior still requires
              the same semantic contracts, concrete inputs, numerical tolerances and source
              provenance. Performance findings additionally depend on workload and CPU environment.
            </p>
            <p>
              New proposals stay unreviewed until independent replay and human acceptance. The
              official roadmap's reviewed progress is unaffected by indexing an observation.
            </p>
          </section>
        </>
      )}
    </main>
  );
}
