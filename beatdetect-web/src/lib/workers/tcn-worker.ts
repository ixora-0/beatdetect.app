import * as ort from 'onnxruntime-web';
import { fetchBinaryCached, getModelURL } from '$lib/utils/model-registry';

self.onmessage = async function (e) {
  const { mel: mel_data, flux: flux_data, n_mels } = e.data;

  const url = await getModelURL('beat_model');
  const bytes = await fetchBinaryCached(url);
  const session = await ort.InferenceSession.create(bytes);

  const T = mel_data.length / n_mels;
  const mel = new ort.Tensor('float32', mel_data, [1, n_mels, T]);
  const flux = new ort.Tensor('float32', flux_data, [1, T]);
  const outputs = await session.run({ mel, flux });

  self.postMessage({
    type: 'complete',
    probs: await outputs.probs.getData()
  });
};
