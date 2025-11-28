import * as ort from 'onnxruntime-web';
import modelURL from '@models/onnx/beattcn.onnx?url'; // if fail run save_onnx first
import type { TCNInput } from '$lib/types/worker-types';

self.onmessage = async function (e) {
  let { mel: mel_data, flux: flux_data, n_mels }: TCNInput = e.data;

  const session = await ort.InferenceSession.create(modelURL);
  const T = mel_data.length / n_mels;
  const mel = new ort.Tensor('float32', mel_data, [1, n_mels, T]);
  const flux = new ort.Tensor('float32', flux_data, [1, T]);
  const outputs = await session.run({ mel, flux });

  self.postMessage({
    type: 'complete',
    probs: await outputs.probs.getData()
  });
};
