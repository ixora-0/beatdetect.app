import pathlib

DOWNLOADS_PATH = pathlib.Path("data/downloads")

ANNOTATIONS_GITHUB_USER = "CPJKU"
ANNOTATIONS_GITHUB_REPO = "beat_this_annotations"
ANNOTATIONS_GITHUB_BRANCH = "main"
ANNOTATIONS_RAW_PATH = pathlib.Path("data/raw/annotations")
ANNOTATIONS_PROCESSED_PATH = pathlib.Path("data/processed/annotations")
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
