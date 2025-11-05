<script lang="ts">
  import IconFileAudioRegular from 'phosphor-icons-svelte/IconFileAudioRegular.svelte';
  import IconTrashBold from 'phosphor-icons-svelte/IconTrashBold.svelte';
  import IconWarningRegular from 'phosphor-icons-svelte/IconWarningRegular.svelte';

  import { FileUpload } from '@skeletonlabs/skeleton-svelte';
  import { Toast, createToaster } from '@skeletonlabs/skeleton-svelte';

  const toaster = createToaster({
    placement: 'top-end'
  });
  interface Props {
    onFileUploaded?: (file: File) => void;
    onFileClear?: () => void;
  }
  let { onFileUploaded, onFileClear }: Props = $props();

  let uploadedFile: File | null = $state(null);
  function onFileAccept(details: { files: File[] }) {
    if (details.files.length > 0) {
      onFileUploaded?.(details.files[0]);
      uploadedFile = details.files[0];
    }
  }
  function onFileReject(details: { files: { file: File; errors: string[] }[] }) {
    toaster.error({
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
      class="btn-icon preset-outlined transition-all hover:preset-filled-error-500 hover:shadow-lg hover:shadow-error-500/50"
      title="Clear file"
      aria-label="Clear file"
    >
      <IconTrashBold class="size-6" />
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
