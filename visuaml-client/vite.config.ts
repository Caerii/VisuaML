import react from '@vitejs/plugin-react-swc';

// https://vite.dev/config/
export default {
  plugins: [react()],
  server: {
    proxy: {
      // string shorthand: '/foo' -> 'http://localhost:4567/foo'
      // '/api': 'http://localhost:8787',
      // With options:
      '/api': {
        target: 'http://localhost:8787',
        changeOrigin: true,
        // remove /api prefix if your API server doesn't expect it
        // rewrite: (path) => path.replace(/^\/api/, '')
      },
    },
  },
};
