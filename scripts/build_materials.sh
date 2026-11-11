#!/bin/sh
# Compiles materials/*.mat into public/materials/*.filamat with matc from
# Filament's release of the same version as the Filament.js scripts/vendor.mjs
# fetches (1.80.0; a .filamat only loads in the Filament.js it was built for):
#   https://github.com/google/filament/releases/download/v1.80.0/filament-v1.80.0-linux.tgz
# Usage: scripts/build_materials.sh [path/to/matc] [name ...]  (default: matc on PATH, every material)
set -e
cd "$(dirname "$0")/.."
MATC=${1:-matc}
[ $# -gt 0 ] && shift
names=${*:-$(cd materials && ls *.mat | sed 's/\.mat$//')}
for n in $names; do
  echo "matc $n"
  "$MATC" -a opengl -p mobile -o "public/materials/$n.filamat" "materials/$n.mat"
done
