// Should be similar to beatdetect-model/config_loader.py

export interface RawPaths {
  readonly annotations: string;
  readonly spectrograms: string;
}

export interface InterimPaths {
  readonly annotations: string;
}

export interface ProcessedPaths {
  readonly splits_info: string;
  readonly annotations: string;
  readonly spectral_flux: string;
  readonly datasets_info: string;
}

export interface DataPaths {
  readonly raw: RawPaths;
  readonly interim: InterimPaths;
  readonly processed: ProcessedPaths;
}

export interface Paths {
  readonly models: string;
  readonly downloads: string;
  readonly transitions: string;
  readonly init_dist: string;
  readonly data: DataPaths;
}

export interface DownloadsAnnotations {
  readonly github_user: string;
  readonly github_repo: string;
  readonly github_branch: string;
}

export interface DownloadsSpectrograms {
  readonly url_template: string;
}

export interface Downloads {
  readonly rc_api_url: string;
  readonly datasets: readonly string[];
  readonly annotations: DownloadsAnnotations;
  readonly spectrograms: DownloadsSpectrograms;
}

export interface HypersConfig {
  readonly learning_rate: number;
  readonly dropout: number;
  readonly kernel_size: number;
  readonly channels: readonly number[];
  readonly dilations: readonly number[];
}

export interface PostConfig {
  readonly tempo_bins: readonly number[];
  readonly time_signatures: readonly number[];
  readonly pi: number;
  readonly lambda1: number;
  readonly lambda2: number;
}

export interface SpectrogramConfig {
  readonly sample_rate: number;
  readonly n_fft: number;
  readonly f_min: number;
  readonly f_max: number;
  readonly n_mels: number;
  readonly mel_scale: string;
  readonly hop_length: number;
  readonly n_stft: number; // calculated
  readonly fps: number; // calculated
}

export interface TrainingConfig {
  readonly train_ratio: number;
  readonly val_ratio: number;
  readonly batch_size: number;
  readonly max_batch_size: number;
  readonly test_ratio: number; // calculated
}

export interface Config {
  readonly random_seed: number;
  readonly paths: Paths;
  readonly downloads: Downloads;
  readonly spectrogram: SpectrogramConfig;
  readonly training: TrainingConfig;
  readonly hypers: HypersConfig;
  readonly post: PostConfig;
}

/** Config without calculated fields */
export interface TomlConfig {
  readonly random_seed: number;
  readonly paths: Paths;
  readonly downloads: Downloads;
  readonly spectrogram: Omit<SpectrogramConfig, 'n_stft' | 'fps'>;
  readonly training: Omit<TrainingConfig, 'test_ratio'>;
  readonly hypers: HypersConfig;
  readonly post: PostConfig;
}
