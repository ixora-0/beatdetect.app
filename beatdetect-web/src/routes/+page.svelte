<script lang="ts">
  import IconArrowRightRegular from 'phosphor-icons-svelte/IconArrowRightRegular.svelte';
  import AudioFileUpload from '$lib/components/AudioFileUpload.svelte';
  import IconWarningRegular from 'phosphor-icons-svelte/IconWarningRegular.svelte';
  import LightSwitch from '$lib/components/LightSwitch.svelte';
  import { Progress } from '@skeletonlabs/skeleton-svelte';
  import { Toast, createToaster } from '@skeletonlabs/skeleton-svelte';
  import WaveSurfer from 'wavesurfer.js';
  import Spectrogram from 'wavesurfer.js/dist/plugins/spectrogram.js';
  import Timeline from 'wavesurfer.js/dist/plugins/timeline.js';
  import Zoom from 'wavesurfer.js/dist/plugins/zoom.js';
  import type { PageProps } from './$types';
  import SpectrogramWorker from './spectrogram-worker.ts?worker';

  let { data }: PageProps = $props();
  const { config } = data;

  let toaster = createToaster({ placement: 'top-end' });
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
        url: url,
        minPxPerSec: 20,
        plugins: [
          Timeline.create({ height: 25, style: { fontSize: '20px' } }),
          Zoom.create({ scale: 0.1, maxZoom: 200 })
        ]
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
  async function calculateSpectrogram(): Promise<number[][] | null> {
    if (uploadedFile === null) {
      toaster.error({ title: "Can't process file because it is not loaded." });
      return null;
    }

    // Read and decode audio
    const audioContext = new window.AudioContext();
    const arrayBuffer = await uploadedFile.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Resample to config's sample_rate if necessary
    let resampledBuffer = audioBuffer;
    if (audioBuffer.sampleRate !== config.spectrogram.sample_rate) {
      const offlineContext = new OfflineAudioContext(
        audioBuffer.numberOfChannels,
        audioBuffer.duration * config.spectrogram.sample_rate,
        config.spectrogram.sample_rate
      );
      const source = offlineContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(offlineContext.destination);
      source.start();
      resampledBuffer = await offlineContext.startRendering();
    }

    // Convert to mono
    const mono = new Float32Array(resampledBuffer.length);
    for (let i = 0; i < resampledBuffer.length; i++) {
      let sum = 0;
      // Average channels
      for (let channel = 0; channel < resampledBuffer.numberOfChannels; channel++) {
        sum += resampledBuffer.getChannelData(channel)[i];
      }
      mono[i] = sum / resampledBuffer.numberOfChannels;
    }

    const worker = new SpectrogramWorker();
    return new Promise((resolve, reject) => {
      worker.onmessage = (e) => {
        if (e.data.type == 'progress') {
          progress = e.data.progress;
        } else if (e.data.type == 'complete') {
          progress = null;
          const { melSpect } = e.data;
          worker.terminate();
          resolve(melSpect);
        }
      };

      worker.onerror = (e) => {
        worker.terminate();
        toaster.error({ title: 'Error while calculating spectrogram.', description: e.error });
        reject();
      };

      worker.postMessage({
        mono: mono,
        config: config.spectrogram
      });
    });
  }

  async function processFile() {
    await calculateSpectrogram();

    // HACK: Recalculating spectrogram here for visualization in wavesurfer
    // Can't get calculated spectrogram from wavesurfer,
    // and frequenciesDataUrl doesn't work
    wavesurfer?.registerPlugin(
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
  }
</script>

<h1>Beat detect app</h1>
<LightSwitch />
<form>
  <!--Not allowing params to change for now-->
  <label class="label">
    <span class="label-text">Sample rate</span>
    <input class="input" type="number" disabled value={config.spectrogram.sample_rate} />
  </label>
  <label class="label">
    <span class="label-text">Hop size</span>
    <input class="input" type="number" disabled value={config.spectrogram.hop_length} />
  </label>
  <label class="label">
    <span class="label-text"># mel bands</span>
    <input class="input" type="number" disabled value={config.spectrogram.n_mels} />
  </label>
</form>

{#if uploadedFile !== null}
  <div
    id="waveform"
    class="w-full {!isWaveformReady ? 'h-32 placeholder animate-pulse' : ''}"
  ></div>
{/if}

<AudioFileUpload
  onFileUploaded={(file) => (uploadedFile = file)}
  onFileClear={() => (uploadedFile = null)}
  {toaster}
/>

{#if isWaveformReady}
  <button class="btn preset-filled" onclick={processFile}
    ><span>Start</span><IconArrowRightRegular class="size-6" /></button
  >
{/if}

{#if progress !== null}
  <div class="mt-4">Processing audio...</div>
  <Progress value={progress}>
    <Progress.Label>{Math.round(progress)}%</Progress.Label>
    <Progress.Track>
      <Progress.Range />
    </Progress.Track>
  </Progress>
{/if}

<Toast.Group {toaster}>
  {#snippet children(toast)}
    <Toast {toast}>
      <IconWarningRegular class="size-8" />
      <Toast.Message>
        <Toast.Title class="flex items-center gap-2">{toast.title}</Toast.Title>
        <Toast.Description>{toast.description}</Toast.Description>
      </Toast.Message>
      <Toast.CloseTrigger />
    </Toast>
  {/snippet}
</Toast.Group>
