<script lang="ts">
  import AudioFileUpload from '$lib/components/AudioFileUpload.svelte';
  import LightSwitch from '$lib/components/LightSwitch.svelte';
  import { Progress } from '@skeletonlabs/skeleton-svelte';

  let progress: number | null = $state(null);
  async function onFileUploaded(file: File) {
    progress = 0;

    // transform audio here
    let start = performance.now();
    function update(now: number) {
      progress = (now - start) / 1000;
      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        progress = null;
      }
    }
    requestAnimationFrame(update);
  }
</script>

<h1>Beat detect app</h1>
<LightSwitch />
<AudioFileUpload {onFileUploaded} />

{#if progress !== null}
  <div class="mt-4">Processing audio...</div>
  <Progress value={progress * 100}>
    <Progress.Label>{Math.round(progress * 100)}%</Progress.Label>
    <Progress.Track>
      <Progress.Range />
    </Progress.Track>
  </Progress>
{/if}
