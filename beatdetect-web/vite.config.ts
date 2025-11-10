import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [tailwindcss(), sveltekit()],
  server: {
    watch: {
      usePolling: true
    }
  },

  // Ignore vite worker_thread warning
  // https://github.com/vitejs/vite/pull/3932
  optimizeDeps: {
    exclude: ['worker_threads']
  }
});
