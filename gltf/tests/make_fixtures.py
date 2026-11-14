#!/usr/bin/env python3
"""Generate GLB fixtures for KHR_animation_pointer gltfpack tests.

The fixtures intentionally use KHR_animation_pointer channels targeting material
PBR factors, texture transforms and KHR_lights_punctual light intensity, plus
negative/regression cases. All buffers are embedded so each fixture is one file.
"""
import json
import os
import struct
import sys

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def pad4(b: bytes) -> bytes:
    return b + b"\x00" * ((4 - len(b) % 4) % 4)


class GlbBuilder:
    def __init__(self):
        self.gltf = {"asset": {"version": "2.0"}}
        self.bin = bytearray()
        self.ext_used = set()
        self.ext_required = set()

    def add_view(self, data: bytes, target=None):
        offset = len(self.bin)
        self.bin += data
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.gltf.setdefault("bufferViews", []).append(view)
        return len(self.gltf["bufferViews"]) - 1

    def add_accessor(self, values, comp_type, atype, normalized=False):
        type_sizes = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
        ct_sizes = {5126: 4, 5123: 2, 5121: 1}
        flat = []
        for v in values:
            flat.extend(v if isinstance(v, (list, tuple)) else [v])
        fmt = {5126: "f", 5123: "H", 5121: "B"}[comp_type]
        raw = b"".join(struct.pack("<" + fmt, float(x) if comp_type == 5126 else int(x)) for x in flat)
        view = self.add_view(raw, target=34962 if atype != "SCALAR" else None)
        count = len(values)
        acc = {"bufferView": view, "componentType": comp_type, "count": count, "type": atype}
        if normalized:
            acc["normalized"] = True
        if comp_type == 5126:
            comps = type_sizes[atype]
            mins = [min(flat[i::comps]) for i in range(comps)] if flat else []
            maxs = [max(flat[i::comps]) for i in range(comps)] if flat else []
            acc["min"] = mins
            acc["max"] = maxs
        self.gltf.setdefault("accessors", []).append(acc)
        return len(self.gltf["accessors"]) - 1

    def use_extension(self, name, required=False):
        self.ext_used.add(name)
        if required:
            self.ext_required.add(name)

    def build(self) -> bytes:
        self.gltf["buffers"] = [{"byteLength": len(self.bin)}]
        self.gltf["extensionsUsed"] = sorted(self.ext_used)
        if self.ext_required:
            self.gltf["extensionsRequired"] = sorted(self.ext_required)
        js = json.dumps(self.gltf, separators=(",", ":")).encode("utf-8")
        js = pad4(js)
        binb = pad4(bytes(self.bin))
        total = 12 + 8 + len(js) + 8 + len(binb)
        out = b"glTF" + struct.pack("<II", 2, total)
        out += struct.pack("<II", len(js), 0x4E4F534A) + js
        out += struct.pack("<II", len(binb), 0x004E4942) + binb
        return out


def triangle_mesh(b: GlbBuilder):
    pos = b.add_accessor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], 5126, "VEC3")
    return pos


def make_mesh(b: GlbBuilder):
    pos = triangle_mesh(b)
    b.gltf["meshes"] = [{"primitives": [{"attributes": {"POSITION": pos}, "mode": 4}]}]
    b.gltf["nodes"] = [{"mesh": 0, "name": "MeshNode"}]
    b.gltf["scenes"] = [{"nodes": [0]}]
    b.gltf["scene"] = 0


def sampler_inputs(b: GlbBuilder):
    return b.add_accessor([0.0, 0.5, 1.0], 5126, "SCALAR")


def pointer_channel(sampler, uri):
    return {
        "sampler": sampler,
        "target": {"extensions": {"KHR_animation_pointer": {"uri": uri}}},
    }


def add_pointer_animation(b: GlbBuilder, channels, outputs, name="Anim"):
    """channels: list of (uri, values, atype); one shared input accessor."""
    inp = sampler_inputs(b)
    samplers = []
    ch = []
    for i, (uri, values, atype) in enumerate(channels):
        out = b.add_accessor(values, 5126, atype)
        samplers.append({"input": inp, "output": out, "interpolation": "LINEAR"})
        ch.append(pointer_channel(i, uri))
    b.gltf["animations"] = [{"name": name, "samplers": samplers, "channels": ch}]
    b.use_extension("KHR_animation_pointer")


def fixture_valid(path):
    b = GlbBuilder()
    make_mesh(b)
    # material 0: metallic-roughness with a texture transform on base color
    b.gltf["materials"] = [{
        "name": "MatA",
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.7, 0.6, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.5,
            "baseColorTexture": {
                "index": 0,
                "extensions": {"KHR_texture_transform": {"offset": [0.1, 0.2], "scale": [1.0, 1.0], "rotation": 0.0}},
            },
        },
    }]
    # minimal texture/image placeholders (uri-less data not needed: texture object without image is tolerated by cgltf)
    b.gltf["textures"] = [{"name": "BaseColor"}]
    b.gltf["meshes"][0]["primitives"][0]["material"] = 0
    # light
    b.use_extension("KHR_lights_punctual")
    b.gltf["extensions"] = {"KHR_lights_punctual": {"lights": [{"type": "point", "intensity": 10.0, "color": [1.0, 1.0, 1.0]}]}}
    b.gltf["nodes"].append({"name": "LightNode", "extensions": {"KHR_lights_punctual": {"light": 0}}})
    b.gltf["scenes"][0]["nodes"].append(1)

    add_pointer_animation(b, [
        ("/materials/0/pbrMetallicRoughness/baseColorFactor",
         [[0.8, 0.7, 0.6, 1.0], [0.2, 0.3, 0.4, 1.0], [0.9, 0.1, 0.5, 1.0]], "VEC4"),
        ("/materials/0/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/offset",
         [[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]], "VEC2"),
        ("/materials/0/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/rotation",
         [0.0, 1.5707963, 3.1415926], "SCALAR"),
        ("/materials/0/pbrMetallicRoughness/metallicFactor",
         [0.0, 0.5, 1.0], "SCALAR"),
        ("/extensions/KHR_lights_punctual/lights/0/intensity",
         [10.0, 20.0, 5.0], "SCALAR"),
        ("/extensions/KHR_lights_punctual/lights/0/color",
         [[1.0, 1.0, 1.0], [1.0, 0.5, 0.2], [0.2, 0.5, 1.0]], "VEC3"),
    ], None)
    open(path, "wb").write(b.build())


def two_materials(b: GlbBuilder):
    pos = triangle_mesh(b)
    b.gltf["meshes"] = [
        {"primitives": [{"attributes": {"POSITION": pos}, "mode": 4, "material": 0}]},
        {"primitives": [{"attributes": {"POSITION": pos}, "mode": 4, "material": 1}]},
    ]
    b.gltf["nodes"] = [{"mesh": 0, "name": "N0"}, {"mesh": 1, "name": "N1"}]
    b.gltf["scenes"] = [{"nodes": [0, 1]}]
    b.gltf["scene"] = 0
    # two structurally identical, unnamed materials -> gltfpack merges them
    b.gltf["materials"] = [
        {"pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.5}},
        {"pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.5}},
    ]


def fixture_dedup(path):
    b = GlbBuilder()
    two_materials(b)
    add_pointer_animation(b, [
        ("/materials/1/pbrMetallicRoughness/baseColorFactor",
         [[0.5, 0.5, 0.5, 1.0], [0.1, 0.2, 0.3, 1.0], [0.9, 0.8, 0.7, 1.0]], "VEC4"),
    ], None)
    open(path, "wb").write(b.build())


def fixture_deleted(path):
    # material 1 exists but is not referenced by any mesh; gltfpack removes unused materials,
    # so the pointer targeting it must be rejected instead of emitting a dangling JSON pointer
    b = GlbBuilder()
    pos = triangle_mesh(b)
    b.gltf["meshes"] = [{"primitives": [{"attributes": {"POSITION": pos}, "mode": 4, "material": 0}]}]
    b.gltf["nodes"] = [{"mesh": 0}]
    b.gltf["scenes"] = [{"nodes": [0]}]
    b.gltf["scene"] = 0
    b.gltf["materials"] = [
        {"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]}},
        {"pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.2, 0.2, 1.0]}},
    ]
    add_pointer_animation(b, [
        ("/materials/1/pbrMetallicRoughness/roughnessFactor", [0.5, 0.7, 0.9], "SCALAR"),
    ], None)
    open(path, "wb").write(b.build())


def fixture_invalid(path):
    b = GlbBuilder()
    make_mesh(b)
    b.gltf["materials"] = [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}]
    add_pointer_animation(b, [
        ("/materials/0/unknownProperty/foo", [1.0, 2.0, 3.0], "SCALAR"),
    ], None)
    open(path, "wb").write(b.build())


def fixture_bad_index(path):
    b = GlbBuilder()
    make_mesh(b)
    b.gltf["materials"] = [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}]
    add_pointer_animation(b, [
        ("/materials/7/pbrMetallicRoughness/metallicFactor", [0.0, 0.5, 1.0], "SCALAR"),
    ], None)
    open(path, "wb").write(b.build())


def fixture_regular(path):
    # no KHR_animation_pointer: ordinary node translation animation
    b = GlbBuilder()
    make_mesh(b)
    inp = sampler_inputs(b)
    out = b.add_accessor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], 5126, "VEC3")
    b.gltf["animations"] = [{
        "name": "Regular",
        "samplers": [{"input": inp, "output": out, "interpolation": "LINEAR"}],
        "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
    }]
    open(path, "wb").write(b.build())


def fixture_mixed(path):
    # one animation mixing a regular node translation channel and a KHR_animation_pointer channel
    b = GlbBuilder()
    make_mesh(b)
    b.gltf["materials"] = [{"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.2, "roughnessFactor": 0.8}}]
    b.gltf["meshes"][0]["primitives"][0]["material"] = 0
    inp = sampler_inputs(b)
    node_out = b.add_accessor([[0, 0, 0], [1, 2, 3], [4, 5, 6]], 5126, "VEC3")
    pointer_out = b.add_accessor([0.2, 0.4, 0.9], 5126, "SCALAR")
    b.gltf["animations"] = [{
        "name": "Mixed",
        "samplers": [
            {"input": inp, "output": node_out, "interpolation": "LINEAR"},
            {"input": inp, "output": pointer_out, "interpolation": "STEP"},
        ],
        "channels": [
            {"sampler": 0, "target": {"node": 0, "path": "translation"}},
            pointer_channel(1, "/materials/0/pbrMetallicRoughness/metallicFactor"),
        ],
    }]
    b.use_extension("KHR_animation_pointer")
    open(path, "wb").write(b.build())


def fixture_cubic(path):
    # CUBICSPLINE interpolation on a pointer channel is not supported and must be rejected, not silently mis-sampled
    b = GlbBuilder()
    make_mesh(b)
    b.gltf["materials"] = [{"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]}}]
    b.gltf["meshes"][0]["primitives"][0]["material"] = 0
    inp = sampler_inputs(b)
    # cubic spline requires 3 vectors per keyframe (in-tangent, value, out-tangent)
    out = b.add_accessor([[0, 0, 0, 1]] * 3 + [[1, 0, 0, 1]] * 3 + [[0, 1, 0, 1]] * 3, 5126, "VEC4")
    b.gltf["animations"] = [{
        "name": "Cubic",
        "samplers": [{"input": inp, "output": out, "interpolation": "CUBICSPLINE"}],
        "channels": [pointer_channel(0, "/materials/0/pbrMetallicRoughness/baseColorFactor")],
    }]
    b.use_extension("KHR_animation_pointer")
    open(path, "wb").write(b.build())


def fixture_component_mismatch(path):
    # a SCALAR output channel targeting a VEC4 material factor must be rejected
    b = GlbBuilder()
    make_mesh(b)
    b.gltf["materials"] = [{"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]}}]
    b.gltf["meshes"][0]["primitives"][0]["material"] = 0
    inp = sampler_inputs(b)
    wrong = b.add_accessor([1.0, 2.0, 3.0], 5126, "SCALAR")
    b.gltf["animations"] = [{
        "name": "Mismatch",
        "samplers": [{"input": inp, "output": wrong}],
        "channels": [pointer_channel(0, "/materials/0/pbrMetallicRoughness/baseColorFactor")],
    }]
    b.use_extension("KHR_animation_pointer")
    open(path, "wb").write(b.build())


def main():
    os.makedirs(DIR, exist_ok=True)
    fixtures = {
        "pointer_valid.glb": fixture_valid,
        "pointer_dedup.glb": fixture_dedup,
        "pointer_deleted.glb": fixture_deleted,
        "pointer_invalid.glb": fixture_invalid,
        "pointer_bad_index.glb": fixture_bad_index,
        "pointer_mixed.glb": fixture_mixed,
        "pointer_cubic.glb": fixture_cubic,
        "pointer_component_mismatch.glb": fixture_component_mismatch,
        "regular.glb": fixture_regular,
    }
    for name, fn in fixtures.items():
        fn(os.path.join(DIR, name))
        print("wrote", name)


if __name__ == "__main__":
    main()
