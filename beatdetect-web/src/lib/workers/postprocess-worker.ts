import { beamSearch, loadArrays } from '$lib/utils/inference';

self.onmessage = async function (e) {
  let { nnOutput, config } = e.data;

  const arrays = await loadArrays();
  const { beats } = beamSearch(config, nnOutput, arrays);
  self.postMessage({
    type: 'complete',
    beats: beats
  });
};
