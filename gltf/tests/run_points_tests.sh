#!/usr/bin/env bash
# Roundtrip tests for indexed POINTS primitives.
#
# Generates GLB test assets, runs gltfpack with quantization/compression/simplification
# option sets, and verifies point counts, bounding boxes, per-attribute quantization
# error, material references, mixed primitives and corrupt-input handling.
# Compressed payloads are additionally decoded with the meshoptimizer JS decoder.
set -euo pipefail

cd "$(dirname "$0")/../.."

if [ ! -x ./gltfpack ]; then
	echo "gltfpack binary not found; build it first: make gltfpack" >&2
	exit 2
fi

python3 -I gltf/tests/gen_points_gltf.py gltf/tests/data
python3 -I gltf/tests/roundtrip_points.py
node gltf/tests/roundtrip_points_compressed.cjs
