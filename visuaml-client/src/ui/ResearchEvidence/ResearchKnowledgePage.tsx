import { useEffect, useMemo, useState } from 'react';
import { parseKnowledgeGraph, relatedKnowledge, type KnowledgeGraph } from './knowledgeModel';
import './ResearchEvidencePage.css';
import './ResearchKnowledgePage.css';

const short = (value: string | null | undefined): string =>
  value ? value.slice(0, 12) : 'Not recorded';

const label = (value: string): string => value.replace(/_/g, ' ');
const percentage = (value: number): string => value.toPrecision(4);

export function ResearchKnowledgePage() {
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    const load = async () => {
      try {
        const response = await fetch('/evidence/knowledge-v36.json', {
          signal: controller.signal,
          cache: 'no-store',
        });
        if (!response.ok) throw new Error('Research knowledge graph is not deployed');
        const raw: unknown = await response.json();
        const validated = parseKnowledgeGraph(raw);
        if (!controller.signal.aborted) {
          setGraph(validated);
          const pinned =
            validated.nodes.find((node) => node.kind === 'pinned_pretrained_independent_replay') ??
            validated.nodes.find((node) => node.kind === 'pinned_pretrained_native_witness');
          setSelected(pinned?.entry_id ?? validated.nodes[0]?.entry_id ?? null);
        }
      } catch (reason) {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Unrecognized research graph');
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };
    void load();
    return () => controller.abort();
  }, []);

  const nodes = useMemo(() => graph?.nodes ?? [], [graph]);
  const visible = useMemo(() => {
    const query = search.trim().toLowerCase();
    return query
      ? nodes.filter((node) =>
          [
            node.label,
            node.kind,
            node.status,
            ...node.op_sequence,
            ...Object.values(node.context),
          ].some((term) => term.toLowerCase().includes(query)),
        )
      : nodes;
  }, [nodes, search]);
  const chosen = useMemo(
    () => visible.find((node) => node.entry_id === selected) ?? visible[0] ?? null,
    [visible, selected],
  );
  const connections = useMemo(
    () => (graph && chosen ? relatedKnowledge(graph, chosen.entry_id) : []),
    [graph, chosen],
  );

  return (
    <main className="research-evidence knowledge-graph">
      <header className="research-evidence-header">
        <nav aria-label="Research navigation">
          <a href="/research/library">Evidence library</a>
          <a href="/research">Research evidence</a>
          <a href="/compiler">Compiler Lab</a>
          <a href="/roadmap">Roadmap</a>
        </nav>
        <p className="research-evidence-eyebrow">VISUAML · CUMULATIVE COMPUTATIONAL KNOWLEDGE</p>
        <h1>Computation has a history.</h1>
        <p className="research-evidence-intro">
          Follow exact checkpoint, semantic, source-code and binary identities across compiler
          experiments. Inspect the old pretrained witness, the newly reconstructed source bundle,
          and independently measured numerical results without conflating them.
        </p>
      </header>

      {loading && <p role="status">Reading the reviewed computational knowledge graph…</p>}
      {error && (
        <section className="research-evidence-unavailable" role="alert">
          <h2>Knowledge graph unavailable on this deployment</h2>
          <p>{error}. Merged source code does not imply the site is deployed.</p>
          <a href="https://github.com/Caerii/VisuaML">Inspect the source repository</a>
        </section>
      )}

      {graph && (
        <>
          <section className="knowledge-statistics" aria-label="Knowledge coverage">
            <div>
              <span className="research-evidence-label">INDEXED STUDIES</span>
              <strong>{graph.counts.studies}</strong>
            </div>
            <div>
              <span className="research-evidence-label">SOURCE ARTIFACTS</span>
              <strong>{graph.counts.source_artifacts}</strong>
            </div>
            <div>
              <span className="research-evidence-label">EXACT IDENTITY RELATIONS</span>
              <strong>{graph.counts.identity_edges}</strong>
            </div>
          </section>

          <section className="research-evidence-section">
            <p className="research-evidence-eyebrow">I. THE KNOWLEDGE GRAPH</p>
            <h2>Different experiments, inspectable relationships.</h2>
            <p>
              An identical checkpoint digest is evidence of shared model bytes, not a proof that an
              optimization is correct for every input or faster on another CPU. Numerical and
              performance claims retain their original finite context.
            </p>
            <label className="knowledge-search">
              Search indexed studies
              <input
                type="search"
                placeholder="Llama, attention, semantic IR, convolution…"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </label>
            <div className="knowledge-layout">
              <div className="knowledge-studies" aria-label="Indexed historical studies">
                {visible.length === 0 && <p>No matching studies.</p>}
                {visible.map((node) => (
                  <button
                    type="button"
                    key={node.entry_id}
                    aria-pressed={chosen?.entry_id === node.entry_id}
                    className={`knowledge-study ${chosen?.entry_id === node.entry_id ? 'active' : ''}`}
                    onClick={() => setSelected(node.entry_id)}
                  >
                    <span className="research-evidence-label">{label(node.kind)}</span>
                    <strong>{node.label}</strong>
                    <span>{label(node.evidence_class)}</span>
                  </button>
                ))}
              </div>
              <article className="knowledge-detail" aria-label="Selected study and lineage">
                {chosen ? (
                  <>
                    <p className="research-evidence-eyebrow">SOURCE-BOUND OBSERVATION</p>
                    <h3>{chosen.label}</h3>
                    <p className="knowledge-status">{label(chosen.status)}</p>
                    <div className="research-evidence-ops">
                      {chosen.op_sequence.map((op, index) => (
                        <span key={index}>
                          <code>{op}</code>
                        </span>
                      ))}
                    </div>
                    <h4>Recorded computational identity</h4>
                    <dl className="knowledge-metadata">
                      {(
                        [
                          ['Checkpoint', chosen.checkpoint_sha256],
                          ['Semantic IR', chosen.semantic_sha256],
                          ['Generated C++', chosen.generated_cpp_sha256],
                          ['Native binary', chosen.native_binary_sha256],
                          ['Model revision', chosen.model_revision],
                        ] as const
                      ).map(([name, value]) => (
                        <div key={name}>
                          <dt>{name}</dt>
                          <dd>
                            <code title={value || ''}>{short(value)}</code>
                          </dd>
                        </div>
                      ))}
                    </dl>
                    <h4>Original experimental measurements</h4>
                    <dl className="knowledge-metadata">
                      {Object.entries(chosen.result).map(([key, value]) => (
                        <div key={key}>
                          <dt>{label(key)}</dt>
                          <dd>{percentage(value)}</dd>
                        </div>
                      ))}
                    </dl>
                    <h4>Exact historical relationships</h4>
                    {connections.length ? (
                      <div className="knowledge-relations">
                        {connections.map(({ node, relation }) => (
                          <button
                            type="button"
                            key={node.entry_id + relation}
                            onClick={() => {
                              setSearch('');
                              setSelected(node.entry_id);
                            }}
                          >
                            <strong>{node.label}</strong>
                            <span>{label(relation)}</span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p>No exact matching identity found in the current corpus.</p>
                    )}
                    <details>
                      <summary>Recorded limitations and provenance</summary>
                      <p>
                        <a href={chosen.workflow_url} target="_blank" rel="noreferrer">
                          Original CI workflow
                        </a>
                      </p>
                      <p>
                        <a
                          href={`https://github.com/Caerii/VisuaML/blob/main/${chosen.source_path}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Reviewed source artifact
                        </a>
                      </p>
                      <ul>
                        {chosen.limitations.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </details>
                  </>
                ) : (
                  <p>Select a study to inspect its source and lineage.</p>
                )}
              </article>
            </div>
          </section>

          <section className="research-evidence-section knowledge-replay">
            <p className="research-evidence-eyebrow">II. INDEPENDENT REPLAY</p>
            <h2>A source hash is not a rerun. A fresh execution is.</h2>
            <p>
              The v35 historical replay plan remains a preflight-only record. A separate v36
              experiment reconstructed the three original source files from independently verified
              Git objects, then performed new pretrained C++17 execution against independent Hugging
              Face and NumPy oracles. The original CI checkout was never established as one Git
              commit.
            </p>
            <div className="knowledge-plan">
              <div>
                <span className="research-evidence-label">ORIGINAL V35 PLAN</span>
                <strong>Source/checkpoint preflight only</strong>
                <p>The old Git checkout identity was not recorded in the original evidence.</p>
                <p>That historical plan was not rewritten or silently marked successful.</p>
                <a
                  href="https://github.com/Caerii/VisuaML/blob/main/docs/V35_REPLAYABLE_KNOWLEDGE_GRAPH.md"
                  target="_blank"
                  rel="noreferrer"
                >
                  Original v35 reproducibility plan
                </a>
              </div>
              {graph.fresh_replays?.[0] ? (
                <div>
                  <span className="research-evidence-label">NEW V36 NUMERICAL RESULT</span>
                  <strong>Independent finite replay passed</strong>
                  <p>
                    The exact reconstructed source-bundle digest is{' '}
                    <code title={graph.fresh_replays[0].source_bundle_sha256}>
                      {short(graph.fresh_replays[0].source_bundle_sha256)}
                    </code>
                    . No single original source Git commit was claimed.
                  </p>
                  <p>
                    A new C++/Hugging Face/NumPy numerical witness is bound to the{' '}
                    <a
                      href={graph.fresh_replays[0].producing_ci_run}
                      target="_blank"
                      rel="noreferrer"
                    >
                      fresh GitHub Actions run
                    </a>
                    . It remains a finite test, not a formal proof.
                  </p>
                  <p>
                    <a
                      href="https://github.com/Caerii/VisuaML/blob/main/docs/V36_PRETRAINED_REPLAY.md"
                      target="_blank"
                      rel="noreferrer"
                    >
                      Read the exact source reconstruction and numerical protocol
                    </a>
                  </p>
                </div>
              ) : (
                <div>
                  <span className="research-evidence-label">FRESH REPLAY</span>
                  <strong>Not yet verified on this snapshot</strong>
                  <p>Use the historical source and model preflight to identify blockers.</p>
                </div>
              )}
            </div>
          </section>
        </>
      )}
    </main>
  );
}
