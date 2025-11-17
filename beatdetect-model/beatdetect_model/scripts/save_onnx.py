import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from ..config_loader import Config, load_config


class OnsetStrength(nn.Module):
    def __init__(
        self, lag: int, max_size: int, center: bool, n_fft: int, hop_length: int
    ):
        super().__init__()
        self.lag = lag
        self.max_size = max_size
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        # Compute reference spectrogram using max filter across time dimension
        # (n_mels, T)
        if self.max_size == 1:
            ref = S
        else:
            pad_size = (self.max_size - 1) // 2
            # Have to input (N, H, W) to pad and max_pool2d
            padded = F.pad(S.unsqueeze(0), (0, 0, pad_size, pad_size), mode="reflect")
            # Use max_pool2d here so don't have to transpose
            ref = F.max_pool2d(
                padded, kernel_size=(self.max_size, 1), stride=1, padding=0
            ).squeeze(0)

        # (n_mels, T - lag)
        onset_env = S[..., self.lag :] - ref[..., : -self.lag]
        onset_env = torch.clamp(onset_env, min=0.0)

        # Mean across frequency bins
        onset_env = torch.mean(onset_env, dim=0)

        # (T - lag) -> (T)
        # Compensate for lag + framing effects by shifting right
        pad_width = self.lag + self.n_fft // (2 * self.hop_length) + 1
        onset_env = F.pad(onset_env, (pad_width, 0), mode="constant", value=0.0)
        onset_env = onset_env[..., : S.shape[1]]  # Trim to match input duration

        return onset_env


class MelSpectAndSpectralFlux(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        f_min: float,
        f_max: float,
        n_mels: int,
        mel_scale: str,
        lag: int,
        max_size: int,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale=mel_scale,
            power=1,
        )

        self.onset = OnsetStrength(
            lag=lag, max_size=max_size, center=True, n_fft=n_fft, hop_length=hop_length
        )

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mel = self.mel_spec(waveform)

        mel_db = torchaudio.transforms.AmplitudeToDB(stype="power")(mel)
        onset = self.onset(mel_db)
        onset = torch.clamp(onset, None, 4)

        return mel, onset


def main(config: Config):
    model = MelSpectAndSpectralFlux(
        sample_rate=config.spectrogram.sample_rate,
        n_fft=config.spectrogram.n_fft,
        hop_length=config.spectrogram.hop_length,
        f_min=config.spectrogram.f_min,
        f_max=config.spectrogram.f_max,
        n_mels=config.spectrogram.n_mels,
        mel_scale=config.spectrogram.mel_scale,
        lag=config.spectral_flux.lag,
        max_size=config.spectral_flux.max_size,
    )

    dummy = torch.randn(config.spectrogram.sample_rate * 2)

    onnx_program = torch.onnx.export(
        model,
        dummy,
        input_names=["waveform"],
        output_names=["mel", "flux"],
        dynamic_shapes={
            "waveform": {0: "num_samples"},
        },
        # report=True,
        dynamo=True,
    )

    p = config.paths.models / "onnx" / "preprocess.onnx"
    print(f"Saving mel spectrogram + spectrogram extractor at {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    onnx_program.save(p)
    print("Done.")


if __name__ == "__main__":
    config = load_config()
    main(config)
