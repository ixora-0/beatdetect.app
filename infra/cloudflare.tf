data "cloudflare_zones" "myzone" {
  filter {
    name = var.domain
  }
}

data "cloudflare_zone" "main" {
  zone_id = var.cloudflare_zone_id
}

resource "cloudflare_record" "site" {
  zone_id = data.cloudflare_zone.main.id
  name    = "@"
  type    = "CNAME"
  value   = "storage.googleapis.com"
  proxied = true
}

resource "cloudflare_worker_script" "gcs_router" {
  name       = "gcs-router"
  account_id = data.cloudflare_zone.main.account_id

  content = <<EOT
async function handleRequest(request) {
  const url = new URL(request.url);
  const path = url.pathname === "/" ? "/index.html" : url.pathname;

  const gcs = "https://storage.googleapis.com/${var.bucket_name}-site" + path;

  return fetch(gcs, {
    headers: {
      "Cache-Control": "public, max-age=3600"
    }
  });
}

addEventListener("fetch", event => {
  event.respondWith(handleRequest(event.request));
});
EOT
}

resource "cloudflare_worker_route" "route" {
  zone_id     = data.cloudflare_zone.main.id
  pattern     = "${var.domain}/*"
  script_name = cloudflare_worker_script.gcs_router.name
}
