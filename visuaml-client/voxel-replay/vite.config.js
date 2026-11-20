import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  build: { rollupOptions: { input: {
    terrain: fileURLToPath(new URL('./index.html', import.meta.url)),
    pocket: fileURLToPath(new URL('./pocket.html', import.meta.url)),
    tape: fileURLToPath(new URL("./pocket-tape.html", import.meta.url)),
    universal: fileURLToPath(new URL("./pocket-universal.html", import.meta.url)),
  } } },
  server: { fs: { allow: [fileURLToPath(new URL('../../', import.meta.url))] } },
});
