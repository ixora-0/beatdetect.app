import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  server: {
    watch: {
      usePolling: true
    }
  },
  resolve: {
    alias: {
      '@configs': path.resolve(__dirname, '../configs')
    }
  },

  // Ignore vite worker_thread warning
  // https://github.com/vitejs/vite/pull/3932
  optimizeDeps: {
    exclude: ['worker_threads']
  }
});
