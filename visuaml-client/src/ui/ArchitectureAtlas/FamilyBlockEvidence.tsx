/** Independently executed synthetic reference-block checks; not model certification. */
import { useEffect, useState } from 'react';

type BlockCertificate = {
  architecture_id: string;
  representative_block: string;
  level: string;
  seed_count: number;
  max_abs_error: number;
  source_kind: string;
  full_model_compiler_executable: false;
  family_checkpoint_certified: false;
  physical_hardware_validated: false;
  witnesses: { seed: number; checked_arrays: number; passed: boolean; input_sha256: string }[];
};

type Evidence = {
  schema: string;
  source_kind: string;
  sha256: string;
  torch_version: string;
  certifications: BlockCertificate[];
};

function hasEvidenceShape(value: unknown): value is Evidence {
  if (!value || typeof value !== 'object') return false;
  const data = value as Partial<Evidence>;
  return (
    data.schema === 'visuaml.representative-block-certification.v12' &&
    data.source_kind === 'synthetic-cpu-fixtures' &&
    typeof data.sha256 === 'string' &&
    Array.isArray(data.certifications) &&
    data.certifications.every(
      (c) =>
        c.seed_count > 0 &&
        c.family_checkpoint_certified === false &&
        c.full_model_compiler_executable === false &&
        c.physical_hardware_validated === false,
    )
  );
}

export function FamilyBlockEvidence({ architectureId }: { architectureId: string }) {
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch('/family_block_certificates.v12.json', { signal: controller.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error(`Reference evidence not available (HTTP ${r.status})`);
        const data: unknown = await r.json();
        if (!hasEvidenceShape(data)) throw new Error('Invalid v12 reference evidence');
        return data;
      })
      .then((data) => setEvidence(data))
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Reference evidence unavailable');
        }
      });
    return () => controller.abort();
  }, []);
  const matching = evidence?.certifications.find((c) => c.architecture_id === architectureId);
  return (
    <div className="atlas-primitives" aria-label="Reference block evidence">
      <h3>v12 independent reference-block witnesses</h3>
      {matching ? (
        <>
          <p>
            <strong>{matching.representative_block}</strong>
          </p>
          <p className="atlas-muted">
            {matching.seed_count} deterministic synthetic CPU seeds; NumPy versus PyTorch; maximum
            absolute error {matching.max_abs_error.toExponential(2)}.
          </p>
          <p className="atlas-muted">
            Status: <strong>{matching.level}</strong>. This verifies only the named mathematical
            block, not a complete checkpoint, general compiler lowering, or any real accelerator
            hardware.
          </p>
          <small>
            Evidence: {evidence?.sha256.slice(0, 18)} · PyTorch {evidence?.torch_version}
          </small>
        </>
      ) : (
        <p className="atlas-muted">
          {error ??
            (evidence
              ? 'No independent reference-block witness registered for this architecture yet.'
              : 'Loading v12 reference evidence…')}
        </p>
      )}
    </div>
  );
}
