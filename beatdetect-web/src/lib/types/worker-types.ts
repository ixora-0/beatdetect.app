import type { Config } from './config';

export interface PreprocessWorkerInput {
  mono: Float32Array<ArrayBuffer>;
}
export interface PreprocessWorkerOutput {
  type: 'complete';
  mel: Float32Array<ArrayBufferLike>;
  flux: Float32Array<ArrayBufferLike>;
}

export interface TCNInput {
  mel: Float32Array<ArrayBufferLike>;
  flux: Float32Array<ArrayBufferLike>;
  n_mels: number;
}

export type NNOutput = Float32Array<ArrayBufferLike>;
export interface TCNOutput {
  type: 'complete';
  probs: NNOutput;
}
export interface PostprocessWorkerInput {
  nnOutput: NNOutput;
  config: Config;
}
export interface PostprocessWorkerOutput {
  type: 'complete';
  beats: number[][];
}
