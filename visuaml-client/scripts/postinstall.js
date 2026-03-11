/**
 * Cross-platform postinstall: optionally run pip install for backend requirements.
 * Skip when SKIP_PYTHON_INSTALL=1. Use VISUAML_PYTHON to choose the Python binary.
 */
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

if (process.env.SKIP_PYTHON_INSTALL === '1') {
  process.exit(0);
}

const python = process.env.VISUAML_PYTHON || 'python';
const root = dirname(dirname(fileURLToPath(import.meta.url)));
const reqPath = join(root, 'backend', 'requirements.txt');

const r = spawnSync(python, ['-m', 'pip', 'install', '-r', reqPath], {
  stdio: 'inherit',
  cwd: root,
  shell: true,
});

process.exit(r.status ?? 1);
