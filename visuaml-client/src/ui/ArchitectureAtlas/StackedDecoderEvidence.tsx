/** Multi-layer executable compiler evidence; no pretrained or accelerator claim. */
import { useEffect, useState } from 'react';

type StackedCase = {
  layers: number;
  seed: number;
  checkpoint_sha256: string;
  witness: {
    passed: boolean;
    chunks: number;
    arrays_checked: number;
    max_abs_error: number;
  };
  generation_matches_reference: true;
  all_layer_cache_matches_reference: true;
};

type StackedCertificate = {
  schema: 'visuaml.stacked-decoder-aot-certificate.v14';
  source_kind: 'synthetic-seeded-checkpoint-fixtures';
  claim_level: string;
  backend: string;
  source_sha256: string;
  sha256: string;
  cases: StackedCase[];
  pretrained_checkpoint_certified: false;
  dynamic_shape_aot_supported: false;
  physical_hardware_validated: false;
  limitations: string[];
};

function isStackedCertificate(value: unknown): value is StackedCertificate {
  if (!value || typeof value !== 'object') return false;
  const document = value as Partial<StackedCertificate>;
  return (
    document.schema === 'visuaml.stacked-decoder-aot-certificate.v14' &&
    document.source_kind === 'synthetic-seeded-checkpoint-fixtures' &&
    typeof document.source_sha256 === 'string' &&
    typeof document.sha256 === 'string' &&
    Array.isArray(document.cases) &&
    document.cases.length > 0 &&
    document.cases.every(
      (item) =>
        Number.isInteger(item.layers) &&
        item.layers >= 1 &&
        item.layers <= 4 &&
        item.witness?.passed === true &&
        Number.isFinite(item.witness.max_abs_error) &&
        item.generation_matches_reference === true &&
        item.all_layer_cache_matches_reference === true,
    ) &&
    document.pretrained_checkpoint_certified === false &&
    document.dynamic_shape_aot_supported === false &&
    document.physical_hardware_validated === false &&
    Array.isArray(document.limitations)
  );
}

export function StackedDecoderEvidence({ architectureId }: { architectureId: string }) {
  const [certificate, setCertificate] = useState<StackedCertificate | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (architectureId !== 'decoder_transformer') return;
    const controller = new AbortController();
    fetch('/stacked_decoder_certificate.v14.json', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`v14 evidence unavailable (${response.status})`);
        const value: unknown = await response.json();
        if (!isStackedCertificate(value)) throw new Error('Invalid v14 evidence schema');
        return value;
      })
      .then((document) => {
        if (!controller.signal.aborted) setCertificate(document);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Unknown evidence error');
        }
      });
    return () => controller.abort();
  }, [architectureId]);

  if (architectureId !== 'decoder_transformer') return null;
  const worst = certificate
    ? Math.max(...certificate.cases.map((item) => item.witness.max_abs_error))
    : 0;
  return (
    <section className="atlas-primitives" aria-label="Multi-layer compiled decoder evidence">
      <h3>v14 · multi-layer compiled decoder</h3>
      {certificate ? (
        <>
          <p>
            Independently checked 1–4-layer GQA/RoPE/SwiGLU stacks, with shape-specialized NumPy-AOT
            execution and per-layer KV-cache verification.
          </p>
          <p className="atlas-muted">
            {certificate.cases.length} depth/seed configurations · worst |Δ|{' '}
            <strong>{worst.toExponential(2)}</strong> · {certificate.backend}.
          </p>
          <div className="atlas-tags">
            {certificate.cases.map((item) => (
              <span className="available" key={`${item.layers}-${item.seed}`}>
                {item.layers} layer{item.layers === 1 ? '' : 's'} · {item.witness.arrays_checked} arrays
              </span>
            ))}
          </div>
          <p className="atlas-muted">
            Canonical checkpoint IO is content-checked, but <strong>no actual pretrained checkpoint,</strong>{' '}
            unrestricted dynamic sequence or physical accelerator is certified.
          </p>
          <details>
            <summary>Source fingerprints and scope boundaries</summary>
            <p className="atlas-muted">Evidence SHA-256: {certificate.sha256}</p>
            <p className="atlas-muted">Semantic source: {certificate.source_sha256}</p>
            {certificate.limitations.map((limit) => (
              <p className="atlas-muted" key={limit}>
                {limit}
              </p>
            ))}
          </details>
        </>
      ) : (
        <p className="atlas-muted">{error ?? 'Loading multi-layer compiler evidence…'}</p>
      )}
    </section>
  );
}
