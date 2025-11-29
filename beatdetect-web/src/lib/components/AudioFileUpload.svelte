<script lang="ts">
  import IconFileAudioRegular from 'phosphor-icons-svelte/IconFileAudioRegular.svelte';
  import IconTrashRegular from 'phosphor-icons-svelte/IconTrashRegular.svelte';

  import { FileUpload } from '@skeletonlabs/skeleton-svelte';
  import { createToaster } from '@skeletonlabs/skeleton-svelte';

  interface Props {
    onFileUploaded?: (file: File) => void;
    onFileClear?: () => void;
    toaster?: ReturnType<typeof createToaster>;
    disableClearFile?: boolean;
  }
  let { onFileUploaded, onFileClear, toaster, disableClearFile }: Props = $props();

  let uploadedFile: File | null = $state(null);
  function onFileAccept(details: { files: File[] }) {
    if (details.files.length > 0) {
      onFileUploaded?.(details.files[0]);
      uploadedFile = details.files[0];
    }
  }
  function onFileReject(details: { files: { file: File; errors: string[] }[] }) {
    toaster?.error({
      title: `Cannot process ${details.files[0].file.name}`,
      description: details.files[0].errors
    });
  }
  function clearFile() {
    uploadedFile = null;
  }
</script>

{#if uploadedFile}
  <div class="flex items-center gap-4 rounded border p-4">
    <IconFileAudioRegular class="size-6" />
    <span>File uploaded: {uploadedFile.name}</span>
    <button
      type="button"
      onclick={() => {
        clearFile();
        if (onFileClear) onFileClear();
      }}
      class="group btn-icon w-auto justify-start gap-0 preset-outlined shadow-error-500/50 transition-all hover:gap-1 hover:preset-filled-error-500 hover:shadow-lg"
      title="Clear file"
      aria-label="Clear file"
      disabled={disableClearFile}
    >
      <IconTrashRegular class="size-6" />
      <span class="max-w-0 overflow-hidden whitespace-nowrap transition-all group-hover:max-w-xs">
        Clear file
      </span>
    </button>
  </div>
{:else}
  <FileUpload accept="audio/*" maxFiles={1} {onFileAccept} {onFileReject}>
    <FileUpload.Dropzone>
      <IconFileAudioRegular class="size-10" />
      <span>Select file or drag here.</span>
      <FileUpload.Trigger>Browse Files</FileUpload.Trigger>
      <FileUpload.HiddenInput />
    </FileUpload.Dropzone>
  </FileUpload>
{/if}
