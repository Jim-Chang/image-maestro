#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/.."

docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d
