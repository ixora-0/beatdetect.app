import tomllib
from pathlib import Path
from string import Template

from pydantic import BaseModel, field_validator


class RawPaths(BaseModel):
    annotations: Path
    spectrograms: Path


class InterimPaths(BaseModel):
    annotations: Path


class ProcessedPaths(BaseModel):
    splits_info: Path
    annotations: Path
    spectral_flux: Path


class DataPaths(BaseModel):
    raw: RawPaths
    interim: InterimPaths
    processed: ProcessedPaths


class Paths(BaseModel):
    models: Path
    downloads: Path
    data: DataPaths


class DownloadsAnnotations(BaseModel):
    github_user: str
    github_repo: str
    github_branch: str


class DownloadsSpectrograms(BaseModel):
    url_template: Template

    model_config = {
        "arbitrary_types_allowed": True  # allow Template
    }

    @field_validator("url_template", mode="before")
    def make_template(cls, v):
        from string import Template

        if isinstance(v, Template):
            return v
        return Template(v)


class Downloads(BaseModel):
    rc_api_url: str
    datasets: list[str]
    annotations: DownloadsAnnotations
    spectrograms: DownloadsSpectrograms


class SpectrogramConfig(BaseModel):
    sample_rate: int
    n_fft: int
    f_min: int
    f_max: int
    n_mels: int
    mel_scale: str
    hop_length: int

    @property
    def n_stft(self) -> int:
        return (self.n_fft // 2) + 1

    @property
    def fps(self) -> float:
        return self.sample_rate / self.hop_length


class TrainingConfig(BaseModel):
    train_ratio: float
    val_ratio: float
    batch_size: int

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio


class HypersConfig(BaseModel):
    learning_rate: float
    dropout: float
    kernel_size: int
    channels: list[int]
    dilations: list[int]


class Config(BaseModel):
    random_seed: int
    paths: Paths
    downloads: Downloads
    spectrogram: SpectrogramConfig
    training: TrainingConfig
    hypers: HypersConfig


def load_config(path: str | Path = "configs/dev.toml") -> Config:
    path = Path(path)
    with path.open("rb") as f:
        cfg_dict = tomllib.load(f)
    return Config(**cfg_dict)
