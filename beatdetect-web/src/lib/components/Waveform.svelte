<script lang="ts">
  import IconPlayRegular from 'phosphor-icons-svelte/IconPlayRegular.svelte';
  import IconPauseRegular from 'phosphor-icons-svelte/IconPauseRegular.svelte';
  import WaveSurfer from 'wavesurfer.js';
  import Spectrogram from 'wavesurfer.js/dist/plugins/spectrogram.js';
  import Timeline from 'wavesurfer.js/dist/plugins/timeline.js';
  import Zoom from 'wavesurfer.js/dist/plugins/zoom.js';
  import type { Config } from '$lib/types/config';
  import { Progress } from '@skeletonlabs/skeleton-svelte';

  interface Props {
    config: Config;
    uploadedFile: File | null;
    onReadyChange?: (ready: boolean) => void;
  }
  let { config, uploadedFile, onReadyChange }: Props = $props();

  let wavesurfer: WaveSurfer | null = null;
  let ready = $state(false);
  $effect(() => {
    if (onReadyChange) onReadyChange(ready);
  });
  let playing = $state(false);

  $effect(() => {
    if (wavesurfer) {
      wavesurfer.destroy();
      wavesurfer = null;
    }
    ready = false;

    if (uploadedFile) {
      const url = URL.createObjectURL(uploadedFile);

      wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#4F4A85',
        progressColor: '#383351',
        url: url,
        minPxPerSec: 20,
        plugins: [
          Timeline.create({ height: 25, style: { fontSize: '20px' } }),
          Zoom.create({ scale: 0.1, maxZoom: 200 })
        ]
      });

      wavesurfer.on('play', () => {
        playing = true;
      });
      wavesurfer.on('pause', () => {
        playing = false;
      });

      wavesurfer.on('interaction', () => {
        wavesurfer?.play();
      });

      wavesurfer.once('ready', () => {
        URL.revokeObjectURL(url);
        ready = true;
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

  export function addSpectrogram(): Promise<void> {
    // HACK: Recalculating spectrogram here for visualization in wavesurfer
    // Can't get calculated spectrogram from wavesurfer,
    // and frequenciesDataUrl doesn't work
    const spectrogramPlugin = wavesurfer?.registerPlugin(
      Spectrogram.create({
        sampleRate: config.spectrogram.sample_rate,
        // This creates white background
        // frequencyMax: config.spectrogram.f_max,
        // frequencyMin: config.spectrogram.f_min,
        fftSamples: config.spectrogram.n_fft,
        useWebWorker: true,
        scale: 'mel',
        labels: true
      })
    );
    return new Promise((resolve) => spectrogramPlugin?.once('ready', () => resolve()));
  }
</script>

{#if uploadedFile !== null}
  <div
    id="waveform"
    class="w-full {!ready
      ? 'flex h-32 placeholder animate-pulse items-center justify-center gap-4'
      : ''}"
  >
    {#if !ready}
      <Progress class="w-fit" value={null}>
        <Progress.Circle style="--size: 24px; --thickness: 4px;">
          <Progress.CircleTrack />
          <Progress.CircleRange />
        </Progress.Circle>
      </Progress>
      <span>Loading waveform...</span>
    {:else}
      <button class="btn-icon size-6" onclick={() => wavesurfer?.playPause()}>
        {#if playing}
          <IconPauseRegular class="h-full w-full" />
        {:else}
          <IconPlayRegular class="h-full w-full" />
        {/if}
      </button>
    {/if}
  </div>
{/if}

<svelte:window
  on:keydown|preventDefault={(e) => {
    if (e.code === 'Space') {
      wavesurfer?.playPause();
    }
  }}
/>
