import { mdsvex } from 'mdsvex';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import path from 'path';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: [vitePreprocess(), mdsvex()],
  kit: {
    adapter: adapter(),
    alias: {
      '@configs': path.resolve(__dirname, '../configs')
    }
  },
  extensions: ['.svelte', '.svx']
};

export default config;
