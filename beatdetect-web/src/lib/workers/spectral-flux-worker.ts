import type { SpectralFluxWorkerInput } from '$lib/types/worker-types';

function extractSpectralFlux(
  melSpect: number[][], // shape [nMels][nFrames], linear power scale
  lag: number,
  maxSize: number
): number[] {
  // implementation based on librosa
  // https://github.com/librosa/librosa/blob/main/librosa/onset.py

  const nMels = melSpect.length;
  const nFrames = melSpect[0].length;

  // Convert to log power
  const amin = 1e-10;
  const logMelspect = Array.from({ length: nMels }, () => new Array(nFrames).fill(0));
  for (let m = 0; m < nMels; m++) {
    for (let t = 0; t < nFrames; t++) {
      const val = Math.max(amin, melSpect[m][t]);
      logMelspect[m][t] = 10 * Math.log10(val) - 10 * Math.log10(1.0);
    }
  }

  // Calculate reference spectrum
  const ref = Array.from({ length: nMels }, () => new Array(nFrames).fill(0));
  const half = Math.floor(maxSize / 2);
  // Apply maximum filter
  for (let m = 0; m < nMels; m++) {
    for (let t = 0; t < nFrames; t++) {
      let maxVal = -Infinity;
      for (let k = -half; k <= half; k++) {
        const idx = m + k;
        if (idx >= 0 && idx < nMels) {
          const val = logMelspect[idx][t];
          if (val > maxVal) maxVal = val;
        }
      }
      ref[m][t] = maxVal;
    }
  }

  // Calculate differences
  const onsetEnv: number[][] = Array.from({ length: nMels }, () => []);
  for (let m = 0; m < nMels; m++) {
    for (let t = lag; t < nFrames; t++) {
      const diff = logMelspect[m][t] - ref[m][t - lag];
      onsetEnv[m].push(Math.max(0, diff));
    }
  }

  // Mean across mel bands
  const onsetEnvMean: number[] = [];
  const newLen = nFrames - lag;
  for (let t = 0; t < newLen; t++) {
    let sum = 0;
    for (let m = 0; m < nMels; m++) sum += onsetEnv[m][t];
    onsetEnvMean.push(sum / nMels);
  }

  // Padding
  const padWidth = lag;
  const padded = new Array(padWidth).fill(0).concat(onsetEnvMean);
  return padded.slice(0, nFrames);
}

self.onmessage = (e) => {
  const { melSpect, config }: SpectralFluxWorkerInput = e.data;

  self.postMessage({
    type: 'complete',
    spectralFlux: extractSpectralFlux(melSpect, config.lag, config.max_size)
  });
};
