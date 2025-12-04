variable "project_id" {}
variable "region" { default = "us-central1" }
variable "location" { default = "US" }
variable "bucket_name" {}
variable "domain" {}
variable "cloudflare_api_token" { sensitive = true }
variable "cloudflare_zone_id" { sensitive = true }
