/** Compiler-executable tiny decoder evidence, separate from family-wide compatibility. */
import { useEffect, useState } from 'react';

type CompiledWitness = {
  seed: number;
  prefill_manifest_sha256: string;
  witness: {
    passed: boolean;
    chunks: number;
    arrays_checked: number;
    max_abs_error: number;
    specializations: number;
  };
  generation_matches_reference: true;
  cache_matches_reference: true;
};

type DecoderCertificate = {
  schema: 'visuaml.decoder-aot-certification.v13';
  architecture_id: 'decoder_transformer';
  claim_level: string;
  source_kind: string;
  backend: string;
  sha256: string;
  cases: CompiledWitness[];
  family_checkpoint_certified: false;
  arbitrary_decoder_supported: false;
  physical_accelerator_validated: false;
  limitations: string[];
};

function isCertificate(value: unknown): value is DecoderCertificate {
  if (typeof value !== 'object' || !value) return false;
  const item = value as Partial<DecoderCertificate>;
  return (
    item.schema === 'visuaml.decoder-aot-certification.v13' &&
    item.architecture_id === 'decoder_transformer' &&
    typeof item.sha256 === 'string' &&
    Array.isArray(item.cases) &&
    item.cases.length > 0 &&
    item.cases.every((c) =>
      c.generation_matches_reference === true &&
      c.cache_matches_reference === true &&
      c.witness?.passed === true &&
      Number.isFinite(c.witness.max_abs_error),
    ) &&
    item.family_checkpoint_certified === false &&
    item.arbitrary_decoder_supported === false &&
    item.physical_accelerator_validated === false &&
    Array.isArray(item.limitations)
  );
}

export function DecoderAOTEvidence({ architectureId }: { architectureId: string }) {
  const [certificate, setCertificate] = useState<DecoderCertificate | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (architectureId !== 'decoder_transformer') return;
    const controller = new AbortController();
    fetch('/decoder_compiler_certificate.v13.json', { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Compiler evidence unavailable (${response.status})`);
        const result: unknown = await response.json();
        if (!isCertificate(result)) throw new Error('Invalid decoder compiler evidence contract');
        return result;
      })
      .then((value) => {
        if (!controller.signal.aborted) setCertificate(value);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Compiler evidence unavailable');
        }
      });
    return () => controller.abort();
  }, [architectureId]);

  if (architectureId !== 'decoder_transformer') return null;
  const worst = certificate ? Math.max(...certificate.cases.map((c) => c.witness.max_abs_error)) : null;
  const arrays = certificate?.cases.reduce((sum, c) => sum + c.witness.arrays_checked, 0);
  return (
    <div className="atlas-primitives" aria-label="Portable decoder compiler evidence">
      <h3>v13 compiled decoder · portable NumPy-AOT</h3>
      {certificate ? (
        <>
          <p>
            Shape-specialized prefill, immutable KV updates and greedy generation executed from
            compiled manifests using {certificate.cases.length} deterministic seeds.
          </p>
          <p className="atlas-muted">
            {arrays} compared outputs and KV tensors · worst absolute difference{' '}
            <strong>{worst?.toExponential(2)}</strong> · no accelerator connected.
          </p>
          <p className="atlas-muted">
            Evidence level: <strong>{certificate.claim_level}</strong>. This does not certify
            pretrained checkpoints, arbitrary decoder architectures, or physical hardware.
          </p>
          <details>
            <summary>Inspect numerical witness and source limitations</summary>
            <p className="atlas-muted">Evidence SHA-256: {certificate.sha256}</p>
            {certificate.cases.map((item) => (
              <p key={item.seed} className="atlas-muted">
                Seed {item.seed}: {item.witness.chunks} chunks,{' '}
                {item.witness.arrays_checked} arrays, AOT manifest{' '}
                {item.prefill_manifest_sha256.slice(0, 16)}…
              </p>
            ))}
            {certificate.limitations.map((limit) => (
              <p key={limit} className="atlas-muted">{limit}</p>
            ))}
          </details>
        </>
      ) : (
        <p className="atlas-muted">{error ?? 'Loading v13 compiler execution evidence…'}</p>
      )}
    </div>
  );
}
