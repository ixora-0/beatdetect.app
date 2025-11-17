export interface PreprocessWorkerInput {
  mono: Float32Array<ArrayBuffer>;
}
export interface PreprocessWorkerOutput {
  type: 'complete';
  mel: Float32Array<ArrayBufferLike>;
  flux: Float32Array<ArrayBufferLike>;
}
