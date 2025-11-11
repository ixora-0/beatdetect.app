import Meyda from 'meyda';

function reflectPad(array: Float32Array, padLeft: number, padRight: number): Float32Array {
  const totalLength = array.length + padLeft + padRight;
  const padded = new Float32Array(totalLength);
  padded.set(array, padLeft);

  // Left
  for (let i = 0; i < padLeft; i++) {
    const reflectIdx = padLeft - i;
    padded[i] = array[Math.min(reflectIdx, array.length - 1)];
  }

  // Right
  for (let i = 0; i < padRight; i++) {
    const reflectIdx = array.length - 2 - i;
    padded[padLeft + array.length + i] = array[Math.max(0, reflectIdx)];
  }

  return padded;
}

self.onmessage = function (e) {
  const { mono, config } = e.data;

  Meyda.bufferSize = config.n_fft;
  Meyda.sampleRate = config.sample_rate;
  Meyda.melBands = config.n_mels;

  const padLength = Math.floor(config.n_fft / 2);
  const y = reflectPad(mono, padLength, padLength);

  const T = Math.ceil((y.length - config.n_fft) / config.hop_length);
  const melSpect: Float64Array[] = Array.from({ length: config.n_mels }, () => new Float64Array(T));

  for (let i = 0; i < T; i++) {
    const frame = i * config.hop_length;

    const window = y.slice(frame, frame + config.n_fft);

    const feature = Meyda.extract('melBands', window);
    if (feature === null) {
      throw new Error('Meyda library returns null.');
    }
    const melBands = feature as number[];

    for (let j = 0; j < config.n_mels; j++) {
      melSpect[j][i] = melBands[j];
    }

    // Send progress updates
    self.postMessage({
      type: 'progress',
      progress: (i / T) * 100
    });
  }

  // Send the final result
  self.postMessage({
    type: 'complete',
    melSpect: melSpect
  });
};
