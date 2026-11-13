#!/usr/bin/env python3
"""Roundtrip verification for indexed POINTS primitives through gltfpack.

Runs gltfpack with several option sets and verifies that:
- indexed points remain indexed POINTS primitives with the same point count
- positions/normals/colors/uvs roundtrip within quantization tolerance
- bounding boxes match (within tolerance)
- material references are preserved
- mixed triangle/points meshes keep both primitives with the triangle untouched
- corrupt indices / mismatched attributes / empty point clouds are rejected safely

Standard library only; run with python3 -I.
"""
import json
import math
import os
import struct
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = os.path.join(SCRIPT_DIR, "out")
GLTFPACK = os.path.join(REPO_ROOT, "gltfpack")

CT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
CT_FMT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, total = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67, "not a GLB: %s" % path
    off = 12
    jsondoc = None
    binchunk = b""
    while off < len(data):
        clen, ctype = struct.unpack("<II", data[off:off + 8])
        chunk = data[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            jsondoc = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            binchunk = chunk
        off += 8 + clen
    return jsondoc, binchunk


def accessor_raw_bytes(doc, blob, acc_index):
    """Raw encoded bytes of an accessor (follows bufferView byteOffset/stride)."""
    acc = doc["accessors"][acc_index]
    view = doc["bufferViews"][acc["bufferView"]]
    csize = CT_SIZE[acc["componentType"]]
    comps = TYPE_COMPONENTS[acc["type"]]
    stride = view.get("byteStride", csize * comps)
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    out = bytearray()
    for i in range(acc["count"]):
        out += blob[offset + i * stride:offset + i * stride + csize * comps]
    return bytes(out)


def decode_accessor(doc, blob, acc_index):
    """Decode an accessor to a list of tuples of floats (normalized types un-normalized)."""
    acc = doc["accessors"][acc_index]
    raw = accessor_raw_bytes(doc, blob, acc_index)
    ctype = acc["componentType"]
    comps = TYPE_COMPONENTS[acc["type"]]
    fmt = CT_FMT[ctype]
    count = acc["count"]
    values = struct.unpack("<%d%s" % (count * comps, fmt), raw)
    rows = [values[i * comps:(i + 1) * comps] for i in range(count)]
    if acc.get("normalized"):
        if ctype == 5121:
            rows = [tuple(v / 255.0 for v in r) for r in rows]
        elif ctype == 5123:
            rows = [tuple(v / 65535.0 for v in r) for r in rows]
        elif ctype == 5120:
            rows = [tuple(max(v / 127.0, -1.0) for v in r) for r in rows]
        elif ctype == 5122:
            rows = [tuple(max(v / 32767.0, -1.0) for v in r) for r in rows]
    return rows


def find_node_for_mesh(doc, mesh_index):
    for node in doc["nodes"]:
        if node.get("mesh") == mesh_index:
            return node
    return None


def dequantize_positions(doc, rows, mesh_index):
    """Apply KHR_mesh_quantization node translation/scale to integer position rows."""
    node = find_node_for_mesh(doc, mesh_index)
    if not node:
        return rows
    t = node.get("translation", [0.0, 0.0, 0.0])
    s = node.get("scale", [1.0, 1.0, 1.0])
    return [tuple(t[k] + s[k] * r[k] for k in range(3)) for r in rows]


def view_is_compressed(doc, acc_index):
    acc = doc["accessors"][acc_index]
    view = doc["bufferViews"][acc["bufferView"]]
    ext = view.get("extensions", {})
    return "EXT_meshopt_compression" in ext or "KHR_meshopt_compression" in ext


def collect_points(doc, blob, mesh):
    """Return (point_rows, vertex_count) for every POINTS primitive of a mesh."""
    result = []
    for prim in mesh["primitives"]:
        if prim.get("mode", 4) != 0:
            continue
        pos_acc = prim["attributes"]["POSITION"]
        vertex_count = doc["accessors"][pos_acc]["count"]
        if "indices" in prim:
            if view_is_compressed(doc, prim["indices"]):
                # compressed payloads are verified by the JS meshopt decoder test
                indices = None
                point_count = doc["accessors"][prim["indices"]]["count"]
            else:
                indices = decode_accessor(doc, blob, prim["indices"])
                point_count = len(indices)
        else:
            indices = [(i,) for i in range(vertex_count)]
            point_count = vertex_count
        result.append((prim, indices, point_count, vertex_count))
    return result


def bounds_of(points):
    mins = [min(p[k] for p in points) for k in range(3)]
    maxs = [max(p[k] for p in points) for k in range(3)]
    return mins, maxs


def referenced_bounds(decoded_positions, indices):
    pts = [decoded_positions[i[0]] for i in indices]
    return bounds_of(pts)


PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  PASS:", msg)
    else:
        FAIL += 1
        print("  FAIL:", msg)


def run_pack(input_name, output_name, extra_args):
    inp = os.path.join(DATA_DIR, input_name)
    out = os.path.join(OUT_DIR, output_name)
    cmd = [GLTFPACK, "-i", inp, "-o", out] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc, out


def test_basic_roundtrip(label, extra_args, pos_tol, expect_compressed):
    print("\n== indexed points roundtrip [%s] ==" % label)
    proc, out = run_pack("indexed_points.glb", "indexed_%s.glb" % label.replace("-", "_"), extra_args)
    print("  rc=%d stderr=%s" % (proc.returncode, proc.stderr.strip().replace("\n", " | ")[:300]))
    check(proc.returncode == 0, "gltfpack exits 0")

    src_doc, src_blob = read_glb(os.path.join(DATA_DIR, "indexed_points.glb"))
    out_doc, out_blob = read_glb(out)

    src_mesh = src_doc["meshes"][0]
    out_mesh = out_doc["meshes"][0]

    src_points = collect_points(src_doc, src_blob, src_mesh)
    out_points = collect_points(out_doc, out_blob, out_mesh)
    check(len(src_points) == 1 and len(out_points) == 1, "exactly one POINTS primitive in and out")

    src_prim, src_indices, src_pcount, src_vcount = src_points[0]
    out_prim, out_indices, out_pcount, out_vcount = out_points[0]

    check("indices" in src_prim and "indices" in out_prim, "indexed points stay indexed")
    check(out_prim.get("mode", 4) == 0, "output primitive mode is POINTS (0)")
    check(src_pcount == out_pcount, "point count preserved (%d == %d)" % (src_pcount, out_pcount))

    # material preserved
    check(src_prim.get("material") == out_prim.get("material") and out_prim.get("material") is not None, "material reference preserved")

    # core attributes present (TEXCOORD is only retained when the material references a texture)
    for attr in ("POSITION", "NORMAL", "COLOR_0"):
        check(attr in out_prim["attributes"], "attribute %s present in output" % attr)

    # compression markers are checked first; compressed payloads are decoded by the JS test instead
    ext_used = any(("EXT_meshopt_compression" in v.get("extensions", {}) or "KHR_meshopt_compression" in v.get("extensions", {}))
                   for v in out_doc.get("bufferViews", []))
    if expect_compressed:
        check(ext_used, "meshopt compression extension present on bufferViews")
        return out_doc
    check(not ext_used, "no compression extension without -c")

    src_pos = dequantize_positions(src_doc, decode_accessor(src_doc, src_blob, src_prim["attributes"]["POSITION"]), 0)
    out_pos = dequantize_positions(out_doc, decode_accessor(out_doc, out_blob, out_prim["attributes"]["POSITION"]), 0)

    src_bounds = referenced_bounds(src_pos, src_indices)
    out_bounds = referenced_bounds(out_pos, out_indices)

    for k in range(3):
        check(abs(src_bounds[0][k] - out_bounds[0][k]) <= pos_tol and abs(src_bounds[1][k] - out_bounds[1][k]) <= pos_tol,
              "bounds axis %d within tolerance %.5f (in %s out %s)" % (k, pos_tol, src_bounds, out_bounds))

    # per-point attributes: spatial sort reorders points, so compare as multisets on rounded positions
    def keyed(rows, indices):
        return [tuple(round(v, 4) for v in rows[i[0]]) for i in indices]

    src_keys = sorted(keyed(src_pos, src_indices))
    out_keys = sorted(keyed(out_pos, out_indices))
    check(len(src_keys) == len(out_keys), "same number of point position keys")
    mismatch = sum(1 for a, b in zip(src_keys, out_keys) if any(abs(a[k] - b[k]) > pos_tol for k in range(3)))
    check(mismatch == 0, "all per-point positions within tolerance %.5f (%d mismatches)" % (pos_tol, mismatch))

    # normals: sign-presence and unit length after oct/snorm quantization
    if "NORMAL" in out_prim["attributes"]:
        src_nrm = decode_accessor(src_doc, src_blob, src_prim["attributes"]["NORMAL"])
        out_nrm = decode_accessor(out_doc, out_blob, out_prim["attributes"]["NORMAL"])
        src_nsign = sorted(1 if src_nrm[i[0]][2] > 0 else -1 for i in src_indices)
        out_nsign = sorted(1 if out_nrm[i[0]][2] > 0 else -1 for i in out_indices)
        check(src_nsign == out_nsign, "normal signs preserved per point")
        bad = sum(1 for r in out_nrm if abs(math.sqrt(sum(v * v for v in r)) - 1.0) > 0.05)
        check(bad == 0, "output normals stay unit length (%d bad)" % bad)

    # colors within 8-bit quantization tolerance
    if "COLOR_0" in out_prim["attributes"]:
        out_col = decode_accessor(out_doc, out_blob, out_prim["attributes"]["COLOR_0"])
        src_col = decode_accessor(src_doc, src_blob, src_prim["attributes"]["COLOR_0"])
        col_tol = 1.5 / 255.0
        src_ckeys = sorted(tuple(round(v, 3) for v in src_col[i[0]]) for i in src_indices)
        out_ckeys = sorted(tuple(round(v, 3) for v in out_col[i[0]]) for i in out_indices)
        cmismatch = 0
        for a, b in zip(src_ckeys, out_ckeys):
            if any(abs(a[k] - b[k]) > col_tol for k in range(4)):
                cmismatch += 1
        check(cmismatch == 0, "all per-point colors within 8-bit tolerance (%d mismatches)" % cmismatch)

    # uvs only survive when retained by filterStreams
    if "TEXCOORD_0" in out_prim["attributes"]:
        out_uv = decode_accessor(out_doc, out_blob, out_prim["attributes"]["TEXCOORD_0"])
        src_uv = decode_accessor(src_doc, src_blob, src_prim["attributes"]["TEXCOORD_0"])
        uv_tol = 1.0 / 4096.0 * 2
        src_ukeys = sorted(tuple(round(v, 4) for v in src_uv[i[0]]) for i in src_indices)
        out_ukeys = sorted(tuple(round(v, 4) for v in out_uv[i[0]]) for i in out_indices)
        umismatch = sum(1 for a, b in zip(src_ukeys, out_ukeys) if any(abs(a[k] - b[k]) > uv_tol for k in range(2)))
        check(umismatch == 0, "all per-point UVs within 12-bit tolerance (%d mismatches)" % umismatch)

    return out_doc


def test_nonindexed_regression():
    print("\n== non-indexed points regression ==")
    proc, out = run_pack("nonindexed_points.glb", "nonindexed_out.glb", [])
    check(proc.returncode == 0, "gltfpack exits 0")
    src_doc, src_blob = read_glb(os.path.join(DATA_DIR, "nonindexed_points.glb"))
    out_doc, out_blob = read_glb(out)
    src_pts = collect_points(src_doc, src_blob, src_doc["meshes"][0])
    out_pts = collect_points(out_doc, out_blob, out_doc["meshes"][0])
    check(len(out_pts) == 1, "one POINTS primitive in output")
    check("indices" not in out_pts[0][0], "non-indexed points stay non-indexed")
    check(src_pts[0][2] == out_pts[0][2], "point count preserved (%d)" % out_pts[0][2])


def test_mixed_primitives():
    print("\n== mixed triangle + indexed points ==")
    proc, out = run_pack("mixed_primitives.glb", "mixed_out.glb", ["-c"])
    check(proc.returncode == 0, "gltfpack exits 0")
    src_doc, src_blob = read_glb(os.path.join(DATA_DIR, "mixed_primitives.glb"))
    out_doc, out_blob = read_glb(out)

    src_mesh = src_doc["meshes"][0]
    out_mesh = out_doc["meshes"][0]
    check(len(out_mesh["primitives"]) == 2, "both primitives retained")

    modes = sorted(p.get("mode", 4) for p in out_mesh["primitives"])
    check(modes == [0, 4], "modes are POINTS(0) and TRIANGLES(4): %s" % modes)

    tri = [p for p in out_mesh["primitives"] if p.get("mode", 4) == 4][0]
    check(out_doc["accessors"][tri["indices"]]["count"] == 3, "triangle primitive keeps 3 indices")
    tri_pos = out_doc["accessors"][tri["attributes"]["POSITION"]]
    check(tri_pos["count"] == 3, "triangle primitive keeps 3 vertices")

    pts = collect_points(out_doc, out_blob, out_mesh)[0]
    src_pts = collect_points(src_doc, src_blob, src_mesh)[0]
    check(pts[2] == src_pts[2], "point count preserved (%d)" % pts[2])
    check("indices" in pts[0], "points primitive stays indexed")


def test_bad_indices():
    print("\n== corrupt out-of-range indices ==")
    proc, out = run_pack("bad_indices.glb", "bad_out.glb", ["-c"])
    # cgltf_validate rejects out-of-range indices structurally; gltfpack must fail cleanly with an error, never crash
    check(proc.returncode != 0, "gltfpack rejects the corrupt file with non-zero exit")
    check("buffer too short" in proc.stderr or "invalid GLTF" in proc.stderr, "emits a validation error: %s" % proc.stderr.strip()[:120])
    check(not os.path.exists(out), "no output file written for corrupt input")


def test_duplicate_indices():
    print("\n== valid range with duplicate point indices ==")
    proc, out = run_pack("duplicate_indices.glb", "dup_out.glb", ["-c"])
    check(proc.returncode == 0, "gltfpack exits 0")
    src_doc, src_blob = read_glb(os.path.join(DATA_DIR, "duplicate_indices.glb"))
    out_doc, out_blob = read_glb(out)
    src_pts = collect_points(src_doc, src_blob, src_doc["meshes"][0])
    out_pts = collect_points(out_doc, out_blob, out_doc["meshes"][0])
    check(src_pts[0][2] == out_pts[0][2], "point count preserved with duplicates (%d)" % out_pts[0][2])
    check("indices" in out_pts[0][0], "stays indexed")
    # index accessor still references only valid vertices
    pos_count = out_doc["accessors"][out_pts[0][0]["attributes"]["POSITION"]]["count"]
    check(out_doc["accessors"][out_pts[0][0]["indices"]]["count"] == 8, "all 8 point indices retained")
    check(pos_count <= 8, "vertex buffer does not grow beyond source vertex count")


def test_attr_mismatch():
    print("\n== attribute count mismatch ==")
    proc, out = run_pack("attr_mismatch.glb", "mismatch_out.glb", [])
    # cgltf_validate requires all primitive attributes to share the position accessor count
    check(proc.returncode != 0, "gltfpack rejects mismatched attribute counts with non-zero exit")
    check("invalid GLTF" in proc.stderr or "buffer too short" in proc.stderr, "emits a validation error: %s" % proc.stderr.strip()[:120])
    check(not os.path.exists(out), "no output file written for mismatched input")


def test_empty_points():
    print("\n== empty points primitive ==")
    proc, out = run_pack("empty_points.glb", "empty_out.glb", [])
    # cgltf_validate rejects primitives whose position accessor count is zero
    check(proc.returncode != 0, "gltfpack rejects the empty primitive with non-zero exit")
    check("invalid GLTF" in proc.stderr or "buffer too short" in proc.stderr, "emits a validation error: %s" % proc.stderr.strip()[:120])
    check(not os.path.exists(out), "no output file written for empty input")


def test_large_points():
    print("\n== large indexed point cloud (32-bit indices) ==")
    proc, out = run_pack("large_indexed_points.glb", "large_out.glb", ["-c"])
    check(proc.returncode == 0, "gltfpack exits 0")
    src_doc, src_blob = read_glb(os.path.join(DATA_DIR, "large_indexed_points.glb"))
    out_doc, out_blob = read_glb(out)
    src_pts = collect_points(src_doc, src_blob, src_doc["meshes"][0])
    out_pts = collect_points(out_doc, out_blob, out_doc["meshes"][0])
    check(src_pts[0][2] == out_pts[0][2], "point count preserved (%d)" % out_pts[0][2])
    check("indices" in out_pts[0][0], "stays indexed")
    idx_acc = out_doc["accessors"][out_pts[0][0]["indices"]]
    check(idx_acc["componentType"] == 5125, "large vertex range uses 32-bit indices")


def test_simplification():
    print("\n== point simplification with -si ==")
    proc, out = run_pack("indexed_points.glb", "simplified_out.glb", ["-si", "0.5"])
    check(proc.returncode == 0, "gltfpack exits 0")
    src_doc, src_blob = read_glb(os.path.join(DATA_DIR, "indexed_points.glb"))
    out_doc, out_blob = read_glb(out)
    src_pts = collect_points(src_doc, src_blob, src_doc["meshes"][0])
    out_pts = collect_points(out_doc, out_blob, out_doc["meshes"][0])
    check(out_pts[0][2] <= src_pts[0][2], "simplification did not increase point count")
    check(out_pts[0][2] > 0, "simplification kept some points")
    check("indices" in out_pts[0][0], "simplified points stay indexed")
    # all output indices in range
    out_pos_count = out_doc["accessors"][out_pts[0][0]["attributes"]["POSITION"]]["count"]
    bad = [i for i, in out_pts[0][1] if i >= out_pos_count]
    check(not bad, "all simplified indices in range")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(GLTFPACK):
        print("gltfpack binary not found at", GLTFPACK)
        return 2

    # default settings: quantization on (14-bit positions), no compression
    test_basic_roundtrip("default", [], pos_tol=10.0 / (1 << 14) * 2, expect_compressed=False)
    # float positions still quantize mantissa to 14 bits via meshopt_quantizeFloat
    test_basic_roundtrip("float", ["-vpf", "-vtf", "-vnf"], pos_tol=1e-3, expect_compressed=False)
    # meshopt compression
    test_basic_roundtrip("compressed", ["-c"], pos_tol=10.0 / (1 << 14) * 2, expect_compressed=True)
    # higher compression with oct/exp filters
    test_basic_roundtrip("compressed_more", ["-cc"], pos_tol=10.0 / (1 << 14) * 2, expect_compressed=True)
    # no quantization at all
    test_basic_roundtrip("no_quantize", ["-noq"], pos_tol=1e-6, expect_compressed=False)

    test_nonindexed_regression()
    test_mixed_primitives()
    test_bad_indices()
    test_duplicate_indices()
    test_attr_mismatch()
    test_empty_points()
    test_large_points()
    test_simplification()

    print("\n==== %d passed, %d failed ====" % (PASS, FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
