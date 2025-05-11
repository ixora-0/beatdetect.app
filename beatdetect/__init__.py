import pathlib
import string

DOWNLOADS_PATH = pathlib.Path("data/downloads")
RC_API_URL = "http://localhost:5572"

ANNOTATIONS_GITHUB_USER = "CPJKU"
ANNOTATIONS_GITHUB_REPO = "beat_this_annotations"
ANNOTATIONS_GITHUB_BRANCH = "main"
ANNOTATIONS_RAW_PATH = pathlib.Path("data/raw/annotations")
ANNOTATIONS_PROCESSED_PATH = pathlib.Path("data/processed/annotations")
ENCODED_BEATS_PATH = pathlib.Path("data/processed/encoded-annotations/")
SPECTRAL_FLUX_PATH = pathlib.Path("data/processed/spectral-flux/")

SPECTROGRAMS_URL_TEMPLATE = string.Template(
    "https://zenodo.org/records/13922116/files/${dataset}.zip?download=1"
)
DATASETS = [
    "asap",
    "ballroom",
    "beatles",
    "candombe",
    "filosax",
    "groove_midi",
    "gtzan",
    "guitarset",
    "hainsworth",
    "harmonix",
    "hjdb",
    "jaah",
    "rwc",
    "simac",
    "smc",
    "tapcorrect",
]
SPECTROGRAMS_RAW_PATH = pathlib.Path("data/raw/spectrograms/")
