/** Fixed-fixture Compiler Lab endpoint; no remote code execution or arbitrary imports. */
import type { FastifyInstance } from 'fastify';
import { execa } from 'execa';
import { z } from 'zod';
import path from 'path';
import { fileURLToPath } from 'url';

const Body = z.object({
  scenario: z.enum(['mlp', 'attention', 'assembly', 'convolution']),
  seed: z.number().int().min(0).max(100_000).default(19),
}).strict();

const cli = process.env.VISUAML_PYTHON || 'python';
const script = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../backend/scripts/compiler_lab_service.py');
const workingDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function registerCompilerLabRoutes(app: FastifyInstance): void {
  app.post('/api/compiler-lab/analyze', async (request, reply) => {
    if (process.env.VISUAML_COMPILER_LAB_ENABLED !== '1') {
      return reply.code(404).send({ message: 'Compiler Lab API is disabled; set VISUAML_COMPILER_LAB_ENABLED=1.' });
    }
    const parsed = Body.safeParse(request.body);
    if (!parsed.success) {
      return reply.code(400).send({ message: 'Unsupported compiler analysis request', issues: parsed.error.issues });
    }
    try {
      const { stdout } = await execa(cli, [script, '--scenario', parsed.data.scenario, '--seed', String(parsed.data.seed)], {
        cwd: workingDirectory,
        timeout: 90_000,
        maxBuffer: 2 * 1024 * 1024,
        env: { ...process.env, PYTHONNOUSERSITE: '1' },
      });
      const result: unknown = JSON.parse(stdout);
      // Not a generic execution endpoint; the fixture subprocess produces a documented contract.
      if (!result || typeof result !== 'object' || !('api_contract' in result)) {
        throw new Error('Missing compiler-lab response contract');
      }
      return reply.send(result);
    } catch (error) {
      app.log.error({ err: error }, 'Compiler Lab analysis failed');
      return reply.code(500).send({ message: 'Compiler Lab analysis failed; inspect server logs.' });
    }
  });
}
