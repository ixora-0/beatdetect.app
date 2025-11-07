<script lang="ts">
  import IconArrowRightRegular from 'phosphor-icons-svelte/IconArrowRightRegular.svelte';
  import IconWarningRegular from 'phosphor-icons-svelte/IconWarningRegular.svelte';
  import AudioFileUpload from '$lib/components/AudioFileUpload.svelte';
  import LightSwitch from '$lib/components/LightSwitch.svelte';
  import Waveform from '$lib/components/Waveform.svelte';
  import Tasks from '$lib/components/Tasks.svelte';
  import { Toast, createToaster } from '@skeletonlabs/skeleton-svelte';
  import type { PageProps } from './$types';
  import SpectrogramWorker from './spectrogram-worker.ts?worker';

  let { data }: PageProps = $props();
  const { config } = data;

  let toaster = createToaster({ placement: 'top-end' });
  let uploadedFile: File | null = $state(null);

  let waveform: Waveform;
  let isWaveformReady = $state(false);
  let tasks: Tasks;

  async function calculateSpectrogram(): Promise<number[][] | null> {
    if (uploadedFile === null) {
      toaster.error({ title: "Can't process file because it is not loaded." });
      return null;
    }

    // Read and decode audio
    const decodeTaskID = tasks.addTask('Decode audio');
    const audioContext = new window.AudioContext();
    const arrayBuffer = await uploadedFile.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    tasks.completeTask(decodeTaskID);

    // Resample to config's sample_rate if necessary
    let resampledBuffer = audioBuffer;
    if (audioBuffer.sampleRate !== config.spectrogram.sample_rate) {
      const resampleTaskID = tasks.addTask('Resampling audio');
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
      tasks.completeTask(resampleTaskID);
    }

    // Convert to mono
    const monoTaskID = tasks.addTask('Convert to mono');
    const mono = new Float32Array(resampledBuffer.length);
    for (let i = 0; i < resampledBuffer.length; i++) {
      let sum = 0;
      // Average channels
      for (let channel = 0; channel < resampledBuffer.numberOfChannels; channel++) {
        sum += resampledBuffer.getChannelData(channel)[i];
      }
      mono[i] = sum / resampledBuffer.numberOfChannels;
    }
    tasks.completeTask(monoTaskID);

    const worker = new SpectrogramWorker();
    return new Promise((resolve, reject) => {
      const melSpectTaskID = tasks.addTask('Extracting spectrogram', 0);

      worker.onmessage = (e) => {
        if (e.data.type == 'progress') {
          tasks.updateTaskProgress(melSpectTaskID, e.data.progress);
        } else if (e.data.type == 'complete') {
          tasks.completeTask(melSpectTaskID);
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
    calculateSpectrogram();

    const visualizeSpectTaskID = tasks.addTask('Creating spectrogram visualization');
    await waveform.addSpectrogram();
    tasks.completeTask(visualizeSpectTaskID);
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
  onFileClear={() => {
    uploadedFile = null;
    tasks.clear();
  }}
  {toaster}
/>

{#if isWaveformReady}
  <button class="btn preset-filled" onclick={processFile}
    ><span>Start</span><IconArrowRightRegular class="size-6" /></button
  >
{/if}

<Tasks bind:this={tasks} />

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
