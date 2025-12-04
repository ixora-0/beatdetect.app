<script>
  import AppBar from './AppBar.svelte';
</script>

<AppBar />

<div class="grid grid-cols-1 md:grid-cols-[auto_auto_1fr]">
  <aside class="sticky h-[calc(100vh-4.5rem)] p-4">
    <slot name="sidebar" />
  </aside>

  <span class="vr"></span>

  <main class="space-y-4 p-4">
    <slot name="main" />
  </main>
</div>
