import * as ort from 'onnxruntime-web';
import { fetchBinaryCached, getModelURL } from '$lib/utils/model-registry';

self.onmessage = async function (e) {
  const { mono } = e.data;

  const url = await getModelURL('preprocess');
  const bytes = await fetchBinaryCached(url);
  const session = await ort.InferenceSession.create(bytes);

  const tensor = new ort.Tensor('float32', mono);
  const outputs = await session.run({ waveform: tensor });

  self.postMessage({
    type: 'complete',
    mel: await outputs.mel.getData(),
    flux: await outputs.flux.getData()
  });
};
