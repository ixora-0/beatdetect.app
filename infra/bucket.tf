resource "google_storage_bucket" "artifacts" {
  name     = var.bucket_name
  location = var.region

  force_destroy = true

  storage_class = "STANDARD"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD"]
    response_header = ["Content-Type"]
    max_age_seconds = 3600
  }
}

resource "google_storage_bucket_iam_binding" "public_read" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"

  members = [
    "allUsers"
  ]
}
