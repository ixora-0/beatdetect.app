import * as ort from 'onnxruntime-web';
import modelURL from '@models/onnx/preprocess.onnx?url'; // if fail run save_onnx first

self.onmessage = async function (e) {
  const { mono } = e.data;

  const session = await ort.InferenceSession.create(modelURL);
  const tensor = new ort.Tensor('float32', mono);
  const outputs = await session.run({ waveform: tensor });

  self.postMessage({
    type: 'complete',
    mel: await outputs.mel.getData(),
    flux: await outputs.flux.getData()
  });
};
