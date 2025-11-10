import { parse } from '@iarna/toml';
import type { SpectrogramConfig, TrainingConfig, Config, TomlConfig } from '$lib/types/config';
import configTomlContent from '@configs/dev.toml?raw';

export default function load_config() {
  // Read and parse TOML file
  const parsed = parse(configTomlContent) as unknown as TomlConfig;
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
    spectral_flux: parsed.spectral_flux,
    training,
    hypers: parsed.hypers,
    post: parsed.post
  };

  return config;
}
