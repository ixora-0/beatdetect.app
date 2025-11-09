import Meyda from 'meyda';

self.onmessage = function (e) {
  const { mono, config } = e.data;

  Meyda.bufferSize = config.n_fft;
  Meyda.sampleRate = config.sample_rate;
  Meyda.melBands = config.n_mels;

  const melSpect: number[][] = [];

  const end = mono.length - config.n_fft;

  for (let i = 0; i < end; i += config.hop_length) {
    const window = mono.slice(i, i + config.n_fft);

    const melBands = Meyda.extract('melBands', window);
    if (melBands === null) {
      throw new Error('Meyda library returns null.');
    }
    melSpect.push(melBands as number[]);

    // Send progress updates
    self.postMessage({
      type: 'progress',
      progress: (i / end) * 100
    });
  }

  // Send the final result
  self.postMessage({
    type: 'complete',
    melSpect: melSpect
  });
};
