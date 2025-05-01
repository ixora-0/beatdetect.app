import pathlib

ANNOTATIONS_PATH = pathlib.Path("data/annotations")
ANNOTATION_DATASET_PATHS = [p for p in ANNOTATIONS_PATH.iterdir() if p.is_dir()]
