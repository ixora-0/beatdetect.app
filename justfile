mod model "beatdetect-model/"

default:
    just --list

test: model::test
format: model::format
