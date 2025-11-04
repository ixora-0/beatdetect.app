<script lang="ts">
  import { Switch } from '@skeletonlabs/skeleton-svelte';
  import IconSunRegular from 'phosphor-icons-svelte/IconSunRegular.svelte';
  import IconMoonRegular from 'phosphor-icons-svelte/IconMoonRegular.svelte';

  const FALLBACK = 'dark';
  let checked = $state(false);

  $effect(() => {
    const mode = localStorage.getItem('mode') || FALLBACK;
    checked = mode === 'light';
  });

  const onCheckedChange = (event: { checked: boolean }) => {
    const mode = event.checked ? 'light' : 'dark';
    document.documentElement.setAttribute('data-mode', mode);
    localStorage.setItem('mode', mode);
    checked = event.checked;
  };
</script>

<svelte:head>
  <script>
    const mode = localStorage.getItem('mode') || FALLBACK;
    document.documentElement.setAttribute('data-mode', mode);
  </script>
</svelte:head>

<Switch {checked} {onCheckedChange}>
  <Switch.Control>
    <Switch.Thumb>
      <Switch.Context>
        {#snippet children(switch_)}
          {#if switch_().checked}
            <IconSunRegular class="size-3" />
          {:else}
            <IconMoonRegular class="size-3" />
          {/if}
        {/snippet}
      </Switch.Context>
    </Switch.Thumb>
  </Switch.Control>
  <Switch.HiddenInput />
</Switch>
