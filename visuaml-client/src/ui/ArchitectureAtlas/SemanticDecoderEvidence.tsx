/** Live, allowlisted IR inspection. Evidence remains synthetic and CPU-only. */
import { useEffect, useState } from 'react';

type Operation = {
  name: string;
  opcode: string;
  operands: string[];
  result: string | null;
  shape: Array<number | string> | null;
  effect: string;
  cell: string | null;
};
type Inspection = {
  schema: 'visuaml.semantic-inspection.v19';
  source: string;
  fixture: { layers: number; seed: number; batch: number };
  semantic_sha256: string;
  checkpoint_sha256: string;
  operations: Operation[];
  memory: { peak_logical_bytes: number; peak_logical_values: number; limitations: string[] };
  verification: { passed: true; worst_abs_error: number; reference: string };
  limitations: string[];
};

function valid(value: unknown): value is Inspection {
  if (!value || typeof value !== 'object') return false;
  const doc = value as Partial<Inspection>;
  return (
    doc.schema === 'visuaml.semantic-inspection.v19' &&
    Array.isArray(doc.operations) &&
    doc.operations.every(
      (op) =>
        typeof op.name === 'string' && typeof op.opcode === 'string' &&
        Array.isArray(op.operands) && typeof op.effect === 'string',
    ) &&
    doc.verification?.passed === true &&
    Number.isFinite(doc.verification.worst_abs_error) &&
    doc.memory != null &&
    Number.isFinite(doc.memory.peak_logical_bytes) &&
    Array.isArray(doc.limitations)
  );
}

export function SemanticDecoderEvidence({ architectureId }: { architectureId: string }) {
  const [layers, setLayers] = useState(2);
  const [report, setReport] = useState<Inspection | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (architectureId !== 'decoder_transformer') return;
    const controller = new AbortController();
    setReport(null);
    setError(null);
    fetch('/api/compiler-lab/semantic-inspect?layers=' + layers + '&seed=19&batch=1', {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(
          response.status === 404
            ? 'Start the local API with VISUAML_COMPILER_LAB_ENABLED=1 to inspect the IR.'
            : 'Semantic inspection failed: HTTP ' + response.status,
        );
        const value: unknown = await response.json();
        if (!valid(value)) throw new Error('Invalid semantic inspection report');
        return value;
      })
      .then((value) => {
        if (!controller.signal.aborted) setReport(value);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Inspection unavailable');
        }
      });
    return () => controller.abort();
  }, [architectureId, layers]);

  if (architectureId !== 'decoder_transformer') return null;
  return (
    <section className="atlas-primitives" aria-label="Executable semantic decoder inspection">
      <h3>v19 · live semantic compiler inspection</h3>
      <p className="atlas-muted">
        Only trusted, fixed synthetic NumPy CPU decoder fixtures. The logical-memory analysis is
        not a measured device-memory result or pretrained-checkpoint certification.
      </p>
      <label htmlFor="semantic-layers">Decoder layers </label>
      <select
        id="semantic-layers"
        value={layers}
        onChange={(event) => setLayers(Number(event.target.value))}
      >
        {[1, 2, 3, 4].map((count) => (
          <option key={count} value={count}>{count}</option>
        ))}
      </select>
      {report ? (
        <>
          <div className="atlas-tags">
            <span className="available">{report.operations.length} typed operations</span>
            <span className="available">
              max |Δ| {report.verification.worst_abs_error.toExponential(2)}
            </span>
            <span className="available">
              {report.memory.peak_logical_bytes.toLocaleString()} logical bytes at peak
            </span>
          </div>
          <p className="atlas-muted">
            Oracle: {report.verification.reference}. Peak logical SSA values:{' '}
            {report.memory.peak_logical_values}.
          </p>
          <details>
            <summary>Inspect operations, shapes, dependencies and effects</summary>
            <div style={{ maxHeight: 340, overflow: 'auto', fontSize: 12 }}>
              {report.operations.map((op, index) => (
                <div
                  key={op.name}
                  style={{
                    padding: '7px 0', borderBottom: '1px solid #293d50',
                    overflowWrap: 'anywhere',
                  }}
                >
                  <strong>{index + 1}. {op.name}</strong> · {op.opcode}
                  {op.effect !== 'pure' && ' · ' + op.effect + '(' + op.cell + ')'}
                  <div className="atlas-muted">
                    {op.operands.join(' · ')} → {op.result ?? 'state'}{' '}
                    {op.shape?.join(' × ')}
                  </div>
                </div>
              ))}
            </div>
          </details>
          <details>
            <summary>Fingerprints and scope boundaries</summary>
            <p className="atlas-muted">Semantic SHA-256: {report.semantic_sha256}</p>
            <p className="atlas-muted">Checkpoint SHA-256: {report.checkpoint_sha256}</p>
            {report.limitations.map((item) => (
              <p className="atlas-muted" key={item}>{item}</p>
            ))}
          </details>
        </>
      ) : (
        <p className="atlas-muted" role="status">
          {error ?? 'Loading fixed local compiler witness…'}
        </p>
      )}
    </section>
  );
}
