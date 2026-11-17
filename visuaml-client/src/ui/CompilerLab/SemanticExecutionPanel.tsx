import { useEffect, useMemo, useState } from 'react';

type Operation = {
  name: string;
  opcode: string;
  operands: string[];
  result: string | null;
  shape: Array<string | number> | null;
  effect: string;
  cell: string | null;
};
type InspectionReport = {
  schema: 'visuaml.semantic-inspection.v19';
  fixture: { layers: number; seed: number; batch: number };
  semantic_sha256: string;
  checkpoint_sha256: string;
  operations: Operation[];
  memory: { peak_logical_bytes: number; peak_logical_values: number };
  verification: { passed: boolean; worst_abs_error: number; reference: string };
  limitations: string[];
};

function admit(value: unknown): value is InspectionReport {
  if (!value || typeof value !== 'object') return false;
  const report = value as Partial<InspectionReport>;
  return (
    report.schema === 'visuaml.semantic-inspection.v19' &&
    Array.isArray(report.operations) &&
    report.operations.length > 0 &&
    report.operations.every(
      (op) =>
        typeof op.name === 'string' &&
        typeof op.opcode === 'string' &&
        typeof op.effect === 'string' &&
        Array.isArray(op.operands),
    ) &&
    typeof report.semantic_sha256 === 'string' &&
    typeof report.checkpoint_sha256 === 'string' &&
    Number.isFinite(report.memory?.peak_logical_bytes) &&
    report.verification?.passed === true &&
    Number.isFinite(report.verification.worst_abs_error) &&
    Array.isArray(report.limitations)
  );
}

/** Inspect a fixed local fixture, not arbitrary user-supplied model code. */
export function SemanticExecutionPanel() {
  const [open, setOpen] = useState(false);
  const [layers, setLayers] = useState(2);
  const [report, setReport] = useState<InspectionReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [showEffectsOnly, setShowEffectsOnly] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setReport(null);
    setError(null);
    setLoading(true);
    fetch('/api/compiler-lab/semantic-inspect?layers=' + layers + '&seed=19&batch=1', {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(
            response.status === 404
              ? 'Enable VISUAML_COMPILER_LAB_ENABLED=1 on the local API service.'
              : 'Semantic inspection returned HTTP ' + String(response.status),
          );
        }
        const result: unknown = await response.json();
        if (!admit(result)) throw new Error('The compiler returned an invalid evidence report.');
        return result;
      })
      .then((data) => {
        if (!controller.signal.aborted) {
          setReport(data);
          setSelected(data.operations[0]?.name ?? null);
        }
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) {
          setError(cause instanceof Error ? cause.message : 'The inspection is unavailable.');
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [open, layers]);

  const histogram = useMemo(() => {
    if (!report) return [];
    const counts = new Map<string, number>();
    for (const op of report.operations) {
      counts.set(op.opcode, (counts.get(op.opcode) ?? 0) + 1);
    }
    return [...counts].sort((a, b) => b[1] - a[1]);
  }, [report]);

  const filtered = useMemo(
    () =>
      report?.operations.filter((op) => {
        const inEffect = !showEffectsOnly || op.effect !== 'pure';
        const needle = query.trim().toLowerCase();
        return (
          inEffect &&
          (!needle ||
            [op.name, op.opcode, op.cell ?? '', ...op.operands].some((value) =>
              value.toLowerCase().includes(needle),
            ))
        );
      }) ?? [],
    [report, query, showEffectsOnly],
  );
  const current = report?.operations.find((item) => item.name === selected);
  const effectCount = report?.operations.filter((op) => op.effect !== 'pure').length ?? 0;

  return (
    <section className="semantic-studio" aria-labelledby="semantic-studio-title">
      <div className="semantic-studio-intro">
        <div>
          <span className="semantic-studio-kicker">THE EXECUTABLE REPRESENTATION</span>
          <h2 id="semantic-studio-title">Study the computation beneath the graph.</h2>
          <p>
            The visual graph tells us what connects. The semantic execution ledger reveals which
            operations run, the state they read and write, and the numerical witnesses supporting an
            admitted compiler program.
          </p>
        </div>
        <button
          type="button"
          className="semantic-studio-open"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          {open ? 'Close the inspector' : 'Inspect the semantic program ↗'}
        </button>
      </div>

      {open && (
        <div className="semantic-studio-body">
          <div className="semantic-studio-controls">
            <label>
              Decoder depth
              <select value={layers} onChange={(event) => setLayers(Number(event.target.value))}>
                {[1, 2, 3, 4].map((count) => (
                  <option value={count} key={count}>
                    {count} layers
                  </option>
                ))}
              </select>
            </label>
            <label>
              Find an operation
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                type="search"
                placeholder="e.g. attention, rope, stage_key"
              />
            </label>
            <label className="semantic-studio-check">
              <input
                type="checkbox"
                checked={showEffectsOnly}
                onChange={(event) => setShowEffectsOnly(event.target.checked)}
              />
              State effects only
            </label>
          </div>
          {loading && (
            <p role="status" className="semantic-studio-notice">
              Checking a bounded, seeded CPU decoder…
            </p>
          )}
          {error && (
            <p role="alert" className="semantic-studio-notice">
              {error} This inspector accepts fixed synthetic fixtures only; the public
              <a href="/roadmap"> research roadmap</a> remains available without Python.
            </p>
          )}
          {report && (
            <>
              <div className="semantic-studio-metrics">
                <div>
                  <span>Typed operations</span>
                  <strong>{report.operations.length}</strong>
                </div>
                <div>
                  <span>Explicit state effects</span>
                  <strong>{effectCount}</strong>
                </div>
                <div>
                  <span>Logical peak storage</span>
                  <strong>{(report.memory.peak_logical_bytes / 1024).toFixed(1)} KiB</strong>
                </div>
                <div>
                  <span>Worst witness error</span>
                  <strong>{report.verification.worst_abs_error.toExponential(2)}</strong>
                </div>
              </div>
              <div className="semantic-studio-layout">
                <div
                  className="semantic-studio-operations"
                  role="group"
                  aria-label="Operations in verified execution order"
                >
                  <p className="semantic-studio-small">
                    {filtered.length} of {report.operations.length} instructions
                  </p>
                  {filtered.map((op, index) => (
                    <button
                      type="button"
                      key={op.name}
                      className={'semantic-studio-op' + (selected === op.name ? ' active' : '')}
                      aria-pressed={selected === op.name}
                      onClick={() => setSelected(op.name)}
                    >
                      <span>{String(index + 1).padStart(2, '0')}</span>
                      <strong>{op.name}</strong>
                      <em>{op.opcode}</em>
                      {op.effect !== 'pure' && <b>{op.effect}</b>}
                    </button>
                  ))}
                  {filtered.length === 0 && (
                    <p className="semantic-studio-small">No matching operations.</p>
                  )}
                </div>
                <div className="semantic-studio-detail">
                  {current ? (
                    <>
                      <span className="semantic-studio-kicker">OPERATION DOSSIER</span>
                      <h3>{current.name}</h3>
                      <p>
                        {current.effect === 'pure'
                          ? 'Pure tensor operation—no declared state mutation.'
                          : 'Ordered state effect. Its position is part of the decoder contract.'}
                      </p>
                      <dl>
                        <dt>Opcode</dt>
                        <dd>{current.opcode}</dd>
                        <dt>Arguments</dt>
                        <dd>{current.operands.join(' · ')}</dd>
                        <dt>Result</dt>
                        <dd>{current.result ?? 'Effect only'}</dd>
                        <dt>Tensor shape</dt>
                        <dd>{current.shape?.join(' × ') ?? 'No tensor result'}</dd>
                        <dt>State cell</dt>
                        <dd>{current.cell ?? 'None'}</dd>
                      </dl>
                    </>
                  ) : (
                    <p>Select a typed operation to inspect its contract.</p>
                  )}
                  <h4>Operation distribution</h4>
                  <div className="semantic-studio-histogram">
                    {histogram.slice(0, 9).map(([name, count]) => (
                      <div key={name}>
                        <span>{name}</span>
                        <div>
                          <span
                            style={{
                              width: String((100 * count) / report.operations.length) + '%',
                            }}
                          />
                        </div>
                        <b>{count}</b>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="semantic-studio-provenance">
                <div>
                  <strong>Numerical reference: {report.verification.reference}</strong>
                  <p>
                    Semantic SHA-256: {report.semantic_sha256}
                    <br />
                    Checkpoint SHA-256: {report.checkpoint_sha256}
                  </p>
                  <p>
                    Fixed synthetic CPU input only. No pretrained checkpoint, native GPU or optical
                    device is certified by this view.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(report, null, 2)], {
                      type: 'application/json',
                    });
                    const url = URL.createObjectURL(blob);
                    const anchor = document.createElement('a');
                    anchor.href = url;
                    anchor.download = 'visuaml-semantic-evidence.json';
                    anchor.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  Export evidence JSON ↗
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}
