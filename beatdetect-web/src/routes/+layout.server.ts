import path from 'path';
import { readFileSync } from 'fs';
import { parse } from '@iarna/toml';
import type { SpectrogramConfig, TrainingConfig, Config, TomlConfig } from '$lib/types/config';
import type { LayoutServerLoad } from './$types';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const load: LayoutServerLoad = async () => {
  // Read and parse TOML file
  // NOTE: careful about path in prod
  const DEFAULT_PATH = path.resolve(__dirname, '../../../configs/dev.toml');
  const tomlContent = readFileSync(DEFAULT_PATH, 'utf-8');
  const parsed = parse(tomlContent) as unknown as TomlConfig;
  // Add calculated fields
  const spectrogram: SpectrogramConfig = {
    ...parsed.spectrogram,
    n_stft: Math.floor(parsed.spectrogram.n_fft / 2) + 1,
    fps: parsed.spectrogram.sample_rate / parsed.spectrogram.hop_length
  };
  const training: TrainingConfig = {
    ...parsed.training,
    test_ratio: 1.0 - parsed.training.train_ratio - parsed.training.val_ratio
  };
  const config: Config = {
    random_seed: parsed.random_seed,
    paths: parsed.paths,
    downloads: parsed.downloads,
    spectrogram,
    training,
    hypers: parsed.hypers,
    post: parsed.post
  };

  return { config };
};
