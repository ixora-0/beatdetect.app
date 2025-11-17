import json
import sys

import librosa
import numpy as np
import torch
import torchaudio

from beatdetect_model.config_loader import Config, load_config


def main(config: Config, file_path: str):
    y, _sr = librosa.load(file_path, sr=config.spectrogram.sample_rate)

    waveform = torch.tensor(y)
    mel_spect = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.spectrogram.sample_rate,
        n_fft=config.spectrogram.n_fft,
        hop_length=config.spectrogram.hop_length,
        f_min=config.spectrogram.f_min,
        f_max=config.spectrogram.f_max,
        n_mels=config.spectrogram.n_mels,
        mel_scale=config.spectrogram.mel_scale,
        power=1.0,
    )(waveform).numpy()

    log_mel = librosa.power_to_db(mel_spect)
    flux = librosa.onset.onset_strength(
        S=log_mel,
        sr=config.spectrogram.sample_rate,
        hop_length=config.spectrogram.hop_length,
        lag=config.spectral_flux.lag,
        max_size=config.spectral_flux.max_size,
    )
    flux = np.clip(flux, None, 4)
    print(json.dumps({"melSpect": mel_spect.tolist(), "flux": flux.tolist()}))


if __name__ == "__main__":
    config = load_config()
    file_path = sys.argv[1]
    main(config, file_path)
