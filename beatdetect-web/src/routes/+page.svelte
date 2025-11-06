<script lang="ts">
  import IconArrowRightRegular from 'phosphor-icons-svelte/IconArrowRightRegular.svelte';
  import AudioFileUpload from '$lib/components/AudioFileUpload.svelte';
  import LightSwitch from '$lib/components/LightSwitch.svelte';
  import { Progress } from '@skeletonlabs/skeleton-svelte';
  import WaveSurfer from 'wavesurfer.js';
  import type { PageProps } from './$types';

  let { data }: PageProps = $props();
  const { config } = data;
  console.log(config);

  let uploadedFile: File | null = $state(null);
  let wavesurfer: WaveSurfer | null = null;
  let isWaveformReady = $state(false);

  $effect(() => {
    if (wavesurfer) {
      wavesurfer.destroy();
      wavesurfer = null;
    }
    isWaveformReady = false;

    if (uploadedFile) {
      const url = URL.createObjectURL(uploadedFile);

      wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#4F4A85',
        progressColor: '#383351',
        url: url
      });

      wavesurfer.on('interaction', () => {
        wavesurfer?.play();
      });

      wavesurfer.on('ready', () => {
        URL.revokeObjectURL(url);
        isWaveformReady = true;
      });
    }

    // Cleanup function runs when effect re-runs or component unmounts
    return () => {
      if (wavesurfer) {
        wavesurfer.destroy();
        wavesurfer = null;
      }
    };
  });

  let progress: number | null = $state(null);
  async function processFile() {
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
{#if uploadedFile !== null}
  <div
    id="waveform"
    class="h-32 w-full {!isWaveformReady ? 'placeholder animate-pulse' : ''}"
  ></div>
{/if}

<AudioFileUpload
  onFileUploaded={(file) => (uploadedFile = file)}
  onFileClear={() => (uploadedFile = null)}
/>

{#if isWaveformReady}
  <button class="btn preset-filled" onclick={processFile}
    ><span>Start</span><IconArrowRightRegular class="size-6" /></button
  >
{/if}

{#if progress !== null}
  <div class="mt-4">Processing audio...</div>
  <Progress value={progress * 100}>
    <Progress.Label>{Math.round(progress * 100)}%</Progress.Label>
    <Progress.Track>
      <Progress.Range />
    </Progress.Track>
  </Progress>
{/if}
