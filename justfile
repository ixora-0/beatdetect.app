mod model "beatdetect-model/"
mod web "beatdetect-web/"

default:
    just --list

test: model::test web::test
format: model::format web::format
lint: model::lint web::lint
