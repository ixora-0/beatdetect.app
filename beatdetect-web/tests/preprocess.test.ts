import { describe, expect, test } from 'bun:test';
import { fileURLToPath, spawnSync } from 'bun';
import { resampleBuffer, toMono } from '$lib/utils/audio';
import load_config from '$lib/utils/config-loader';
import { Worker } from 'worker_threads';
import type { PreprocessWorkerInput, PreprocessWorkerOutput } from '$lib/types/worker-types';
import decode from 'audio-decode';

// can't use web audio api here, have to use library
async function decodeAudioPath(filePath: string) {
  const file = Bun.file(filePath);
  const buffer = await file.arrayBuffer();
  const audioBuffer = await decode(buffer);
  return audioBuffer;
}

async function runWorker<I, O>(worker: Worker, input: I): Promise<O> {
  return new Promise((resolve, reject) => {
    worker.on('message', (data) => {
      if (data.type === 'complete') {
        worker.terminate();
        resolve(data);
      }
    });

    worker.on('error', (error) => {
      reject(error);
    });

    worker.on('exit', (code) => {
      if (code !== 0) {
        reject(new Error(`Worker stopped with exit code ${code}`));
      }
    });

    worker.postMessage(input);
  });
}
function isClose(
  a: Float32Array<ArrayBufferLike>,
  b: Float32Array<ArrayBufferLike>,
  atol: number,
  rtol: number
): boolean {
  if (a.length != b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    const diff = Math.abs(a[i] - b[i]);
    const tol = atol + rtol * Math.abs(b[i]);
    if (diff > tol) {
      console.log(diff);
      return false;
    }
  }
  return true;
}

const config = load_config();
function toAbs(rp: string): string {
  return fileURLToPath(new URL(rp, import.meta.url));
}
const cwd = toAbs('../../beatdetect-model');
const testAudioPaths = ['1.wav', '2.wav', '3.wav'];

describe('web implementation matches python', () => {
  test('check python (if fail try running uv sync)', () => {
    const pythonCheck = spawnSync(['python', '--version'], { cwd });
    expect(pythonCheck.success && pythonCheck.exitCode === 0).toBeTrue();
    const librosaCheck = spawnSync(['python', '-c', 'import librosa'], { cwd });
    expect(librosaCheck.success && librosaCheck.exitCode === 0).toBeTrue();
  });
  test.each(testAudioPaths)(
    'numerical comparison for %p',
    async (testAudioPath) => {
      testAudioPath = toAbs(testAudioPath);
      const audioBuffer = await decodeAudioPath(testAudioPath);
      const resampledBuffer = await resampleBuffer(audioBuffer, config.spectrogram.sample_rate);
      const mono = await toMono(resampledBuffer);

      const { mel, flux } = await runWorker<PreprocessWorkerInput, PreprocessWorkerOutput>(
        new Worker(toAbs('../src/lib/workers/preprocess-worker.ts')),
        { mono }
      );

      const refResult = Bun.spawnSync(['python', 'beatdetect_model/reference.py', testAudioPath], {
        cwd
      });
      interface RawRefResult {
        melSpect: number[][];
        flux: number[];
      }
      console.error(refResult.stderr.toString());
      expect(refResult.success && refResult.exitCode === 0).toBeTrue();

      const { melSpect: rawRefMel, flux: rawRefFlux }: RawRefResult = JSON.parse(
        refResult.stdout.toString()
      );

      const refMel = new Float32Array(rawRefMel.flat());
      const refFlux = new Float32Array(rawRefFlux);

      // not strict tolerance here
      // there's a difference in decoded waveform between audio-decode and librosa.load
      expect(isClose(mel, refMel, 1e-4, 1e-3)).toBeTrue();
      expect(isClose(flux, refFlux, 1e-3, 1e-2)).toBeTrue();
    },
    16000
  );
});
