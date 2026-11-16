/** Guarded v19 inspector: fixed synthetic fixtures, never accepts model code. */
import type { FastifyInstance } from 'fastify';
import { execa } from 'execa';
import { z } from 'zod';
import path from 'path';
import { fileURLToPath } from 'url';

const Query = z
  .object({
    layers: z.coerce.number().int().min(1).max(4).default(2),
    seed: z.coerce.number().int().min(0).max(100_000).default(19),
    batch: z.coerce.number().int().min(1).max(2).default(1),
  })
  .strict();

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const script = path.join(root, 'backend', 'scripts', 'semantic_inspection_v19.py');
let active = 0;

export function registerSemanticInspectionRoutes(app: FastifyInstance): void {
  app.get('/api/compiler-lab/semantic-inspect', async (request, reply) => {
    if (process.env.VISUAML_COMPILER_LAB_ENABLED !== '1') {
      return reply.code(404).send({ message: 'Compiler Lab API disabled' });
    }
    const parsed = Query.safeParse(request.query);
    if (!parsed.success) {
      return reply.code(400).send({ message: 'Invalid semantic inspection request' });
    }
    if (active >= 2) return reply.code(429).send({ message: 'Inspection capacity reached' });
    active++;
    try {
      const { stdout } = await execa(
        process.env.VISUAML_PYTHON || 'python',
        [
          script,
          '--layers',
          String(parsed.data.layers),
          '--seed',
          String(parsed.data.seed),
          '--batch',
          String(parsed.data.batch),
        ],
        {
          cwd: root,
          timeout: 90_000,
          maxBuffer: 512 * 1024,
          env: { ...process.env, PYTHONNOUSERSITE: '1' },
        },
      );
      const result: unknown = JSON.parse(stdout);
      if (
        !result ||
        typeof result !== 'object' ||
        !('schema' in result) ||
        result.schema !== 'visuaml.semantic-inspection.v19'
      ) {
        throw new Error('Unexpected semantic report schema');
      }
      return reply.send(result);
    } catch (error) {
      app.log.error({ err: error }, 'Semantic inspection failed');
      return reply.code(500).send({ message: 'Semantic inspection failed' });
    } finally {
      active--;
    }
  });
}
