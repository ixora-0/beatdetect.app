<script lang="ts">
  import LightSwitch from '$lib/components/LightSwitch.svelte';
  import { AppBar } from '@skeletonlabs/skeleton-svelte';
  import IconPulseBold from 'phosphor-icons-svelte/IconPulseBold.svelte';
  import IconGithubLogoBold from 'phosphor-icons-svelte/IconGithubLogoBold.svelte';
</script>

<AppBar class="h-18">
  <AppBar.Toolbar class="h-full grid-cols-[auto_1fr_auto]">
    <AppBar.Lead>
      <IconPulseBold class="size-6" />
    </AppBar.Lead>
    <AppBar.Headline>
      <h class="text-2xl">beatdetect.app</h>
    </AppBar.Headline>
    <AppBar.Trail>
      <LightSwitch />
      <a
        href="https://www.github.com/ixora-0/beatdetect.app"
        class="btn-icon btn btn-icon-lg hover:preset-tonal"
        title="github"
        aria-label="github"
      >
        <IconGithubLogoBold />
      </a>
    </AppBar.Trail>
  </AppBar.Toolbar>
</AppBar>
