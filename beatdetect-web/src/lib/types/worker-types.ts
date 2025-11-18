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
export interface TCNOutput {
  type: 'complete';
  probs: Float32Array<ArrayBufferLike>;
}
