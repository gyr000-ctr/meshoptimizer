#!/usr/bin/env python3
"""Round-trip and regression tests for KHR_animation_pointer support in gltfpack.

Run: python3 -I run_tests.py [path-to-gltfpack]

The tests build GLB fixtures (see make_fixtures.py), run gltfpack in several
configurations and inspect the resulting JSON to verify that:
  - material PBR factor, texture transform and light intensity pointer channels survive;
  - JSON pointers are rewritten after material deduplication;
  - invalid/unresolvable pointers and removed targets are rejected with no output;
  - files without the extension keep their previous output.
"""
import json
import os
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")
WORK = os.path.join(HERE, "work")
GLTFPACK = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "gltfpack")

PASSED = 0
FAILED = 0


def read_glb(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"glTF", "not a GLB: %s" % path
    off = 12
    chunks = {}
    while off < len(data):
        length, ctype = struct.unpack_from("<II", data, off)
        off += 8
        chunks[ctype] = data[off:off + length]
        off += length
    return json.loads(chunks[0x4E4F534A].decode("utf-8").rstrip(" \t\r\n\x00")), chunks[0x004E4942]


def run_gltfpack(inp, outp, *extra):
    if os.path.exists(outp):
        os.remove(outp)
    proc = subprocess.run([GLTFPACK, "-i", inp, "-o", outp] + list(extra),
                          capture_output=True, text=True)
    return proc.returncode, proc.stderr


def pointer_uris(doc):
    result = []
    for animation in doc.get("animations", []):
        for channel in animation["channels"]:
            ext = channel.get("target", {}).get("extensions", {}).get("KHR_animation_pointer")
            if ext:
                result.append(ext["uri"])
    return result


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print("PASS: %s" % name)
    else:
        FAILED += 1
        print("FAIL: %s %s" % (name, detail))


def accessor_values(doc, bin_data, accessor_index):
    acc = doc["accessors"][accessor_index]
    view = doc["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    atype = acc["type"]
    comps = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[atype]
    count = acc["count"]
    fmt = {5126: ("f", 4), 5123: ("H", 2), 5121: ("B", 1)}[acc["componentType"]]
    ch, size = fmt
    stride = view.get("byteStride", comps * size)
    values = []
    for i in range(count):
        row = struct.unpack_from("<%d%s" % (comps, ch), bin_data, start + i * stride)
        if acc.get("normalized") and acc["componentType"] != 5126:
            maxv = 65535.0 if acc["componentType"] == 5123 else 255.0
            row = tuple(v / maxv for v in row)
        values.append(list(row) if comps > 1 else row[0])
    return values


def test_valid_roundtrip():
    out = os.path.join(WORK, "valid.glb")
    rc, err = run_gltfpack(os.path.join(FIX, "pointer_valid.glb"), out)
    check("valid: exits 0", rc == 0, err)
    if rc != 0:
        return
    doc, _ = read_glb(out)
    uris = pointer_uris(doc)
    expected = {
        "/materials/0/pbrMetallicRoughness/baseColorFactor",
        "/materials/0/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/offset",
        "/materials/0/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/rotation",
        "/materials/0/pbrMetallicRoughness/metallicFactor",
        "/extensions/KHR_lights_punctual/lights/0/intensity",
        "/extensions/KHR_lights_punctual/lights/0/color",
    }
    check("valid: all six pointer channels preserved", set(uris) == expected, str(uris))
    check("valid: extension declared used", "KHR_animation_pointer" in doc.get("extensionsUsed", []))
    check("valid: extension not required", "KHR_animation_pointer" not in doc.get("extensionsRequired", []))
    # every pointer channel has its own sampler but animation keyframes are not duplicated per channel
    animation = doc["animations"][0]
    check("valid: one sampler per channel", len(animation["samplers"]) == len(animation["channels"]))
    outputs = [s["output"] for s in animation["samplers"]]
    check("valid: no duplicated output accessors", len(outputs) == len(set(outputs)), str(outputs))


def test_keyframe_tolerance():
    # default packing resamples at 30Hz and keeps pointer values in 32-bit floats;
    # compare every output frame against the source piecewise-linear curve at the same timestamp
    src, src_bin = read_glb(os.path.join(FIX, "pointer_valid.glb"))
    out_path = os.path.join(WORK, "tolerance.glb")
    rc, _ = run_gltfpack(os.path.join(FIX, "pointer_valid.glb"), out_path)
    check("tolerance: pack succeeds", rc == 0)
    if rc != 0:
        return
    doc, bin_data = read_glb(out_path)

    def sample_curve(times, values, t):
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]
        for k in range(len(times) - 1):
            if times[k] <= t <= times[k + 1]:
                span = times[k + 1] - times[k]
                f = 0.0 if span == 0 else (t - times[k]) / span
                a = values[k]
                b = values[k + 1]
                if isinstance(a, list):
                    return [x + f * (y - x) for x, y in zip(a, b)]
                return a + f * (b - a)
        return values[-1]

    src_animation = src["animations"][0]
    src_curves = {}
    for channel in src_animation["channels"]:
        uri = channel["target"]["extensions"]["KHR_animation_pointer"]["uri"]
        sampler = src_animation["samplers"][channel["sampler"]]
        times = accessor_values(src, src_bin, sampler["input"])
        values = accessor_values(src, src_bin, sampler["output"])
        src_curves[uri] = (times, values)

    worst = 0.0
    for animation in doc["animations"]:
        for channel in animation["channels"]:
            uri = channel["target"]["extensions"]["KHR_animation_pointer"]["uri"]
            sampler = animation["samplers"][channel["sampler"]]
            out_times = accessor_values(doc, bin_data, sampler["input"])
            out_values = accessor_values(doc, bin_data, sampler["output"])
            ref_times, ref_values = src_curves[uri]
            for t, out in zip(out_times, out_values):
                ref = sample_curve(ref_times, ref_values, t)
                rv = ref if isinstance(ref, list) else [ref]
                ov = out if isinstance(out, list) else [out]
                for a, b in zip(rv, ov):
                    worst = max(worst, abs(a - b))
    check("tolerance: max deviation vs source curve <= 0.01", worst <= 0.01, "worst=%g" % worst)


def test_dedup_rewrite():
    out = os.path.join(WORK, "dedup.glb")
    rc, err = run_gltfpack(os.path.join(FIX, "pointer_dedup.glb"), out)
    check("dedup: exits 0", rc == 0, err)
    if rc != 0:
        return
    doc, _ = read_glb(out)
    uris = pointer_uris(doc)
    # two identical unnamed materials merge into one output material, so /materials/1 must rewrite to /materials/0
    check("dedup: pointer rewritten to merged material", uris == ["/materials/0/pbrMetallicRoughness/baseColorFactor"], str(uris))
    check("dedup: only one material remains", len(doc["materials"]) == 1, str(len(doc["materials"])))


def test_mesh_merge_rewrite():
    out = os.path.join(WORK, "merged.glb")
    rc, err = run_gltfpack(os.path.join(FIX, "pointer_dedup.glb"), out, "-mm")
    check("merge: exits 0", rc == 0, err)
    if rc != 0:
        return
    doc, _ = read_glb(out)
    uris = pointer_uris(doc)
    check("merge: pointer still resolves after mesh merge", all(u.startswith("/materials/0/") for u in uris), str(uris))


def test_invalid_pointer_rejected():
    out = os.path.join(WORK, "invalid.glb")
    rc, err = run_gltfpack(os.path.join(FIX, "pointer_invalid.glb"), out)
    check("invalid: nonzero exit", rc != 0)
    check("invalid: error mentions unsupported property", "unsupported" in err, err)
    check("invalid: no output written", not os.path.exists(out))


def test_bad_index_rejected():
    out = os.path.join(WORK, "bad_index.glb")
    rc, err = run_gltfpack(os.path.join(FIX, "pointer_bad_index.glb"), out)
    check("bad-index: nonzero exit", rc != 0)
    check("bad-index: error mentions missing material", "material" in err, err)
    check("bad-index: no output written", not os.path.exists(out))


def test_deleted_target_rejected():
    out = os.path.join(WORK, "deleted.glb")
    rc, err = run_gltfpack(os.path.join(FIX, "pointer_deleted.glb"), out)
    check("deleted: nonzero exit", rc != 0)
    check("deleted: error mentions removed target", "removed" in err or "KHR_animation_pointer" in err, err)
    check("deleted: no output written", not os.path.exists(out))


def test_compressed_output():
    out = os.path.join(WORK, "compressed.glb")
    rc, err = run_gltfpack(os.path.join(FIX, "pointer_valid.glb"), out, "-cc")
    check("compressed: exits 0", rc == 0, err)
    if rc != 0:
        return
    doc, _ = read_glb(out)
    uris = pointer_uris(doc)
    check("compressed: pointer channels survive compression", len(uris) == 6, str(uris))


def test_regression_no_extension():
    inp = os.path.join(FIX, "regular.glb")
    out_default = os.path.join(WORK, "regular_default.glb")
    out_repeat = os.path.join(WORK, "regular_repeat.glb")
    rc1, err1 = run_gltfpack(inp, out_default)
    rc2, err2 = run_gltfpack(inp, out_repeat)
    check("regression: regular file packs", rc1 == 0 and rc2 == 0, err1 + err2)
    if rc1 == 0 and rc2 == 0:
        d1, _ = read_glb(out_default)
        d2, _ = read_glb(out_repeat)
        check("regression: deterministic output", json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True))
        check("regression: extension not declared", "KHR_animation_pointer" not in d1.get("extensionsUsed", []))
        channels = d1["animations"][0]["channels"]
        check("regression: node target intact", all("node" in c["target"] for c in channels))


def test_mixed_animation():
    out = os.path.join(WORK, "mixed.glb")
    rc, err = run_gltfpack(os.path.join(FIX, "pointer_mixed.glb"), out)
    check("mixed: exits 0", rc == 0, err)
    if rc != 0:
        return
    doc, _ = read_glb(out)
    animation = doc["animations"][0]
    node_channels = [c for c in animation["channels"] if "node" in c["target"]]
    pointer_channels = [c for c in animation["channels"] if "KHR_animation_pointer" in c["target"].get("extensions", {})]
    check("mixed: node channel kept", len(node_channels) == 1)
    check("mixed: pointer channel kept", len(pointer_channels) == 1)
    if pointer_channels:
        uri = pointer_channels[0]["target"]["extensions"]["KHR_animation_pointer"]["uri"]
        check("mixed: pointer uri intact", uri == "/materials/0/pbrMetallicRoughness/metallicFactor", uri)
        sampler = animation["samplers"][pointer_channels[0]["sampler"]]
        check("mixed: step interpolation preserved", sampler.get("interpolation") == "STEP", str(sampler))
    check("mixed: extension declared", "KHR_animation_pointer" in doc.get("extensionsUsed", []))


def test_cubic_rejected():
    out = os.path.join(WORK, "cubic.glb")
    rc, err = run_gltfpack(os.path.join(FIX, "pointer_cubic.glb"), out)
    check("cubic: nonzero exit", rc != 0)
    check("cubic: error mentions CUBICSPLINE", "CUBICSPLINE" in err, err)
    check("cubic: no output written", not os.path.exists(out))


def test_component_mismatch_rejected():
    out = os.path.join(WORK, "mismatch.glb")
    rc, err = run_gltfpack(os.path.join(FIX, "pointer_component_mismatch.glb"), out)
    check("mismatch: nonzero exit", rc != 0)
    check("mismatch: error mentions component count", "component count" in err, err)
    check("mismatch: no output written", not os.path.exists(out))


def main():
    os.makedirs(WORK, exist_ok=True)
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "make_fixtures.py")], check=True)
    for test in (test_valid_roundtrip, test_keyframe_tolerance, test_dedup_rewrite, test_mesh_merge_rewrite,
                 test_invalid_pointer_rejected, test_bad_index_rejected, test_deleted_target_rejected,
                 test_cubic_rejected, test_component_mismatch_rejected, test_compressed_output,
                 test_mixed_animation, test_regression_no_extension):
        test()
    print("\n%d passed, %d failed" % (PASSED, FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
