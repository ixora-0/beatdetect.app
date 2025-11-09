import type { SpectralFluxConfig, SpectrogramConfig } from './config';

export interface SpectrogramWorkerInput {
  mono: Float32Array<ArrayBuffer>;
  config: SpectrogramConfig;
}
export interface SpectrogramWorkerOutput {
  type: 'complete';
  melSpect: number[][];
}

export interface SpectralFluxWorkerInput {
  melSpect: number[][];
  config: SpectralFluxConfig;
}
export interface SpectralFluxWorkerOutput {
  type: 'complete';
  spectralFlux: number[];
}
