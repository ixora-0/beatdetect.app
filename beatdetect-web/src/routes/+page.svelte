<script lang="ts">
  import IconArrowRightRegular from 'phosphor-icons-svelte/IconArrowRightRegular.svelte';
  import IconWarningRegular from 'phosphor-icons-svelte/IconWarningRegular.svelte';
  import AudioFileUpload from '$lib/components/AudioFileUpload.svelte';
  import LightSwitch from '$lib/components/LightSwitch.svelte';
  import Waveform from '$lib/components/Waveform.svelte';
  import Tasks from '$lib/components/Tasks.svelte';
  import { decodeAudio, resampleBuffer, toMono } from '$lib/utils/audio';
  import { Toast, createToaster } from '@skeletonlabs/skeleton-svelte';
  import type { PageProps } from './$types';
  import PreprocessWorker from '$lib/workers/preprocess-worker.ts?worker';
  import type { PreprocessWorkerInput, PreprocessWorkerOutput } from '$lib/types/worker-types';

  let { data }: PageProps = $props();
  const { config } = data;

  let toaster = createToaster({ placement: 'top-end' });
  let uploadedFile: File | null = $state(null);

  let waveform: Waveform;
  let isWaveformReady = $state(false);
  let tasks: Tasks;

  async function extractFeatures() {
    if (uploadedFile === null) {
      toaster.error({ title: "Can't process file because it is not loaded." });
      return null;
    }

    const decodeTaskID = tasks.addTask('Decoding audio file');
    const audioBuffer = await decodeAudio(uploadedFile);
    tasks.completeTask(decodeTaskID);
    const resampleTaskID = tasks.addTask('Resampling audio file');
    const resampledBuffer = await resampleBuffer(audioBuffer, config.spectrogram.sample_rate);
    tasks.completeTask(resampleTaskID);
    const monoTaskID = tasks.addTask('Converting audio file to mono');
    const mono = await toMono(resampledBuffer, (progress) =>
      tasks.updateTaskProgress(monoTaskID, progress)
    );
    tasks.completeTask(monoTaskID);

    async function runWorker<I, O>(
      worker: Worker,
      taskID: string,
      input: I,
      errMsg: string
    ): Promise<O> {
      return new Promise((resolve, reject) => {
        worker.onmessage = (e) => {
          if (e.data.type == 'progress') {
            tasks.updateTaskProgress(taskID, e.data.progress);
          } else if (e.data.type == 'complete') {
            tasks.completeTask(taskID);
            worker.terminate();
            resolve(e.data);
          }
        };
        worker.onerror = (e) => {
          worker.terminate();
          toaster.error({ title: errMsg, description: e.error });
          reject();
        };
        worker.postMessage(input);
      });
    }

    // Extract mel spectrogram
    const { mel, flux } = await runWorker<PreprocessWorkerInput, PreprocessWorkerOutput>(
      new PreprocessWorker(),
      tasks.addTask('Extracting features'),
      { mono },
      'Error while calculating spectrogram.'
    );

    return { mel, flux };
  }

  async function processFile() {
    extractFeatures();

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
