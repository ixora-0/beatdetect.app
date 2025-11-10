import type { SpectralFluxConfig, SpectrogramConfig } from './config';

export interface SpectrogramWorkerInput {
  mono: Float32Array<ArrayBuffer>;
  config: SpectrogramConfig;
}
export interface SpectrogramWorkerOutput {
  type: 'complete';
  melSpect: Float64Array[];
}

export interface SpectralFluxWorkerInput {
  melSpect: Float64Array[];
  config: SpectralFluxConfig;
}
export interface SpectralFluxWorkerOutput {
  type: 'complete';
  spectralFlux: number[];
}
