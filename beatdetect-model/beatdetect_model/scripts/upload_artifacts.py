import hashlib
import json
import sys
from pathlib import Path

from google.cloud import storage

BUCKET_NAME = "beatdetect-app-artifacts"
BUCKET_DIR = "artifacts"

REQUIRED_FILES = [
    "beat_model.onnx",
    "preprocess.onnx",
    "init_dist.npz",
    "transitions.npz",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def upload_and_get_info(bucket, version, name, local_path):
    """Uploads file and returns cloud path + checksum."""
    cloud_path = f"{BUCKET_DIR}/{version}/{local_path.name}"
    blob = bucket.blob(cloud_path)
    blob.upload_from_filename(str(local_path))
    return {
        "path": f"{version}/{local_path.name}",
        "sha256": sha256_file(local_path),
    }


def main(artifacts_dir: Path):
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Directory not found: {artifacts_dir}")

    if not artifacts_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {artifacts_dir}")

    # Infer version from directory name
    version = artifacts_dir.name
    if not version:
        raise ValueError("Could not infer version from directory name")

    print(f"Uploading version {version}")

    # Check all required files exist
    missing_files = []
    artifact_paths = {}

    for filename in REQUIRED_FILES:
        file_path = artifacts_dir / filename
        if not file_path.exists():
            missing_files.append(filename)
        else:
            artifact_paths[file_path.stem] = file_path

    if missing_files:
        raise FileNotFoundError(
            f"Missing required files in {artifacts_dir}: {', '.join(missing_files)}"
        )

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    version_info = {}

    for key, path in artifact_paths.items():
        version_info[key] = upload_and_get_info(bucket, version, key, path)

    # Update global metadata
    global_metadata = {
        "latest": version,
        "files": {k: v["path"] for k, v in version_info.items()},
    }
    bucket.blob(f"{BUCKET_DIR}/metadata.json").upload_from_string(
        json.dumps(global_metadata, indent=2), content_type="application/json"
    )

    print(json.dumps(global_metadata, indent=2))
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    artifacts_directory = Path(sys.argv[1])
    main(artifacts_directory)
