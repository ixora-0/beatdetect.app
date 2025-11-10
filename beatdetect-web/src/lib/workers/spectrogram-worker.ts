import Meyda from 'meyda';

self.onmessage = function (e) {
  const { mono, config } = e.data;

  Meyda.bufferSize = config.n_fft;
  Meyda.sampleRate = config.sample_rate;
  Meyda.melBands = config.n_mels;

  const T = Math.ceil((mono.length - config.n_fft) / config.hop_length);
  const melSpect: Float64Array[] = Array.from({ length: config.n_mels }, () => new Float64Array(T));

  for (let i = 0; i < T; i++) {
    const frame = i * config.hop_length;

    const window = mono.slice(frame, frame + config.n_fft);

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
