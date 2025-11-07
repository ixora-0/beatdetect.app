<script lang="ts">
  import IconArrowRightRegular from 'phosphor-icons-svelte/IconArrowRightRegular.svelte';
  import IconWarningRegular from 'phosphor-icons-svelte/IconWarningRegular.svelte';
  import AudioFileUpload from '$lib/components/AudioFileUpload.svelte';
  import LightSwitch from '$lib/components/LightSwitch.svelte';
  import Waveform from '$lib/components/Waveform.svelte';
  import { Progress } from '@skeletonlabs/skeleton-svelte';
  import { Toast, createToaster } from '@skeletonlabs/skeleton-svelte';
  import type { PageProps } from './$types';
  import SpectrogramWorker from './spectrogram-worker.ts?worker';

  let { data }: PageProps = $props();
  const { config } = data;

  let toaster = createToaster({ placement: 'top-end' });
  let uploadedFile: File | null = $state(null);

  let waveform: Waveform;
  let isWaveformReady = $state(false);

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

  function processFile() {
    calculateSpectrogram();
    waveform.addSpectrogram();
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

<Waveform
  {config}
  {uploadedFile}
  onReadyChange={(ready) => (isWaveformReady = ready)}
  bind:this={waveform}
/>

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
