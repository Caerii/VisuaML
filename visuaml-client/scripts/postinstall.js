/**
 * Cross-platform postinstall: optionally run pip install for backend requirements.
 * By default we skip (so pnpm install is fast). Run pip only when INSTALL_PYTHON_DEPS=1.
 * Use SKIP_PYTHON_INSTALL=1 to force skip; use VISUAML_PYTHON to choose the Python binary.
 */
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const skip = process.env.SKIP_PYTHON_INSTALL === '1' || process.env.INSTALL_PYTHON_DEPS !== '1';
if (skip) {
  console.log('visuaml-client postinstall: Skipping Python deps (fast install).');
  console.log('  To install backend Python deps run: pnpm run install-python-deps   (from repo root)');
  process.exit(0);
}

console.log('visuaml-client postinstall: Installing Python backend deps (torch, transformers, matplotlib, etc.) — may take several minutes...');

const python = process.env.VISUAML_PYTHON || 'python';
const root = dirname(dirname(fileURLToPath(import.meta.url)));
const reqPath = join(root, 'backend', 'requirements.txt');

const r = spawnSync(python, ['-m', 'pip', 'install', '-r', reqPath], {
  stdio: 'inherit',
  cwd: root,
  shell: true,
});

process.exit(r.status ?? 1);
