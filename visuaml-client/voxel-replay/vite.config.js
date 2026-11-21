import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  build: { rollupOptions: { input: {
    terrain: fileURLToPath(new URL('./index.html', import.meta.url)),
    worldStatistics: fileURLToPath(new URL("./pocket-world-statistics.html", import.meta.url)),
    goalPlanning: fileURLToPath(new URL("./pocket-goal-planning.html", import.meta.url)),
    pairedGoals: fileURLToPath(new URL("./pocket-paired-goals.html", import.meta.url)),
    world: fileURLToPath(new URL("./pocket-world-inference.html", import.meta.url)),
    coupled: fileURLToPath(new URL("./pocket-coupled.html", import.meta.url)),
    cellular: fileURLToPath(new URL("./pocket-cellular.html", import.meta.url)),
    pocket: fileURLToPath(new URL('./pocket.html', import.meta.url)),
    tape: fileURLToPath(new URL("./pocket-tape.html", import.meta.url)),
    universal: fileURLToPath(new URL("./pocket-universal.html", import.meta.url)),
    fork: fileURLToPath(new URL("./pocket-tape-fork.html", import.meta.url)),
    continuation: fileURLToPath(new URL("./voxel-call-checkpoint.html", import.meta.url)),
    replay: fileURLToPath(new URL("./replay-viewer.html", import.meta.url)),
  } } },
  server: { proxy: { '/api/pocket-world-reconstruct': 'http://127.0.0.1:4181', '/api/pocket-reconstruct': 'http://127.0.0.1:4181', '/api/voxel-fork-search': 'http://127.0.0.1:4181', '/api/compiler-counterexample': 'http://127.0.0.1:4181' }, fs: { allow: [fileURLToPath(new URL('../../', import.meta.url))] } },
});
