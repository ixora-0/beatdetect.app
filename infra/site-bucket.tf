resource "google_storage_bucket" "site" {
  name          = "${var.bucket_name}-site"
  location      = var.region

  force_destroy = true

  uniform_bucket_level_access = true
  public_access_prevention = "inherited"

  website {
    main_page_suffix = "index.html"
    not_found_page   = "404.html"
  }
}

resource "google_storage_bucket_iam_member" "public_viewer" {
  bucket = google_storage_bucket.site.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}
