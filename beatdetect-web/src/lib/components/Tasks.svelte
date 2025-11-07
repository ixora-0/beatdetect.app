<script lang="ts">
  import { SvelteMap } from 'svelte/reactivity';
  import { Progress } from '@skeletonlabs/skeleton-svelte';

  type TaskStatus = 'running' | 'completed';

  interface Task {
    name: string;

    /** percentage, 0-100 */
    progress: number | null;

    status: TaskStatus;
  }

  let tasks = new SvelteMap<string, Task>();
  const allTasks = $derived(Array.from(tasks.values()));
  // const runningTasks = $derived(allTasks.filter((t) => t.status === 'running'));

  export function addTask(name: string, progress: number | null = null): string {
    const id = crypto.randomUUID();
    // SvelteMap is not deeply reactive, have to have items be states as well
    const task = $state<Task>({
      name,
      progress: progress,
      status: 'running'
    });
    tasks.set(id, task);
    return id;
  }

  export function updateTaskProgress(id: string, progress: number): void {
    const task = tasks.get(id);
    if (!task) return;

    task.progress = Math.min(100, Math.max(0, progress));

    tasks.set(id, task);
  }

  export function completeTask(id: string): void {
    const task = tasks.get(id);
    if (!task) return;

    task.status = 'completed';
    task.progress = 100;
    tasks.set(id, task);
  }

  export function clear(): void {
    tasks.clear();
  }
</script>

<div class="grid grid-cols-[200px_1fr] items-center gap-4">
  {#each allTasks as task}
    <span>{task.name}</span>
    <Progress value={task.progress}>
      {#if task.status == 'completed'}
        <Progress.Label>Done</Progress.Label>
      {:else if task.progress !== null}
        <Progress.Label>{task.progress.toFixed(2)}%</Progress.Label>
      {:else}
        <Progress.Label>In progress</Progress.Label>
      {/if}

      {#if task.status != 'completed'}
        <Progress.Track>
          <Progress.Range />
        </Progress.Track>
      {/if}
    </Progress>
  {/each}
</div>
