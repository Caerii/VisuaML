import { useEffect, useState } from 'react';
import {
  parseEvidenceSnapshot,
  type EvidenceSnapshot,
} from './evidenceModel';
import './ResearchEvidencePage.css';

const shortHash = (value: string) => value.slice(0, 12);
const formatError = (value: number) => value.toExponential(2);
const formatMs = (value: number) => value.toFixed(3);

export function ResearchEvidencePage() {
  const [snapshot, setSnapshot] = useState<EvidenceSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch('/evidence/v33-observation.json', {
          signal: controller.signal,
          cache: 'no-store',
        });
        if (!response.ok) {
          throw new Error('Research evidence is not published on this deployment');
        }
        const data: unknown = await response.json();
        const verifiedShape = parseEvidenceSnapshot(data);
        if (!controller.signal.aborted) setSnapshot(verifiedShape);
      } catch (cause) {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : 'Cannot load research evidence');
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, []);

  return (
    <main className="research-evidence">
      <header className="research-evidence-header">
        <nav aria-label="Research navigation">
          <a href="/compiler">Compiler Lab</a>
          <a href="/roadmap">Public roadmap</a>
          <a href="https://github.com/Caerii/VisuaML" target="_blank" rel="noreferrer">Open-source repository</a>
        </nav>
        <p className="research-evidence-eyebrow">VISUAML · COMPUTATIONAL RESEARCH</p>
        <h1>Evidence is part of the computation.</h1>
        <p className="research-evidence-intro">
          Trace exported mathematical operations into native code, independent numerical
          witnesses, and the actual outcome of proposed transformations. This page shows
          a reviewed CI snapshot—not a continuous live monitor or proof of all inputs.
        </p>
      </header>

      {loading && <p role="status">Reading the published research snapshot…</p>}
      {error && (
        <section className="research-evidence-unavailable" role="alert">
          <h2>Evidence unavailable on this deployment</h2>
          <p>{error}. GitHub source changes alone do not establish that a website has deployed.</p>
          <a href="https://github.com/Caerii/VisuaML/actions">Inspect GitHub verification</a>
        </section>
      )}
      {snapshot && (
        <>
          <section aria-label="Evidence provenance" className="research-evidence-provenance">
            <div>
              <span className="research-evidence-label">REVIEWED SNAPSHOT</span>
              <strong>{snapshot.asOf}</strong>
            </div>
            <div>
              <span className="research-evidence-label">SOURCE REVISION</span>
              <a
                href={`https://github.com/Caerii/VisuaML/tree/${snapshot.sourceCommit}`}
                target="_blank"
                rel="noreferrer"
              >
                <code>{shortHash(snapshot.sourceCommit)}</code>
              </a>
            </div>
            <div>
              <span className="research-evidence-label">REPRODUCIBILITY</span>
              <a href={snapshot.workflowRun} target="_blank" rel="noreferrer">
                Inspect CI and artifact
              </a>
            </div>
          </section>

          <section className="research-evidence-section" aria-labelledby="models-heading">
            <p className="research-evidence-eyebrow">I. ADMITTED EXECUTION</p>
            <h2 id="models-heading">One compiler. Distinct computational structures.</h2>
            <p>
              Both cases use local synthetic weights and the same strict torch.export-to-Semantic-IR
              lowering path. The native implementation was compared with an independent eager
              reference on more than one input.
            </p>
            <div className="research-evidence-grid">
              {snapshot.models.map((model) => (
                <article className="research-evidence-card" key={model.semanticSha256}>
                  <p className="research-evidence-label">{model.kind}</p>
                  <h3>{model.family}</h3>
                  <div className="research-evidence-ops" aria-label="Compiled operator sequence">
                    {model.operations.map((op, index) => (
                      <span key={index}><code>{op}</code></span>
                    ))}
                  </div>
                  <dl className="research-evidence-metrics">
                    <div><dt>Maximum numerical difference</dt>
                      <dd>{formatError(model.maximumAbsoluteError)}</dd></div>
                    <div><dt>Compilation</dt><dd>{formatMs(model.compileMs)} ms</dd></div>
                    <div><dt>Warm invocation median</dt><dd>{formatMs(model.warmMedianMs)} ms</dd></div>
                  </dl>
                  <details>
                    <summary>Inspect source-bound native fingerprints</summary>
                    <p>Semantic IR SHA-256</p><code>{model.semanticSha256}</code>
                    <p>Generated C++ SHA-256</p><code>{model.generatedCppSha256}</code>
                    <p>Compiled binary SHA-256</p><code>{model.binarySha256}</code>
                  </details>
                </article>
              ))}
            </div>
          </section>

          <section className="research-evidence-section" aria-labelledby="transform-heading">
            <p className="research-evidence-eyebrow">II. TRANSFORMATION EXPERIMENT</p>
            <h2 id="transform-heading">A correct rewrite is not automatically a faster rewrite.</h2>
            <div className="research-evidence-experiment">
              <div>
                <span className="research-evidence-label">REWRITE RULE</span>
                <h3><code>{snapshot.transformation.rule}</code></h3>
                <p>{snapshot.transformation.interpretation}</p>
                <p className="research-evidence-technical">
                  Numerical status: <strong>{snapshot.transformation.numericalStatus}</strong>
                </p>
                <details>
                  <summary>Inspect evidence record fingerprint</summary>
                  <code>{snapshot.transformation.evidenceSha256}</code>
                </details>
              </div>
              <div className="research-evidence-ratio">
                <span className="research-evidence-label">OBSERVED BEFORE / AFTER LATENCY</span>
                <strong>{snapshot.transformation.medianRatioBeforeOverAfter.toFixed(3)}×</strong>
                <span>One synthetic CI experiment, not an accepted performance optimization</span>
              </div>
            </div>
          </section>

          <section className="research-evidence-section" aria-labelledby="limits-heading">
            <p className="research-evidence-eyebrow">III. INTERPRETATION</p>
            <h2 id="limits-heading">What the evidence does—and does not—establish.</h2>
            <p>{snapshot.methodology}</p>
            <p className="research-evidence-emphasis">{snapshot.evidenceClass}</p>
            <ul>
              {snapshot.limitations.map((limitation) => (
                <li key={limitation}>{limitation}</li>
              ))}
            </ul>
            <p>
              Research methodology and reproduction instructions live in{' '}
              <a
                href="https://github.com/Caerii/VisuaML/blob/main/docs/V33_EXPORT_AND_EVIDENCE.md"
                target="_blank"
                rel="noreferrer"
              >
                the v33 research protocol
              </a>.
              Accepted roadmap progress remains a separate reviewed decision.
            </p>
          </section>
        </>
      )}
    </main>
  );
}
