#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$project_root/vendor/TRAP-data"

if [[ -d "$target" ]]; then
  printf 'TRAP-data already exists at %s\n' "$target"
  exit 0
fi

mkdir -p "$project_root/vendor"
git clone --depth 1 https://github.com/ricfog/TRAP-data.git "$target"
printf 'Downloaded upstream TRAP data to %s\n' "$target"
