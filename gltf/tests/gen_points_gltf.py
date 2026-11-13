#!/usr/bin/env python3
"""Generate GLB test assets for indexed POINTS primitive roundtrip testing.

All files are written into the data directory passed as the first argument.
Uses only the Python standard library and is intended to be run with python3 -I.
"""
import json
import os
import struct
import sys


class GlbBuilder:
    def __init__(self):
        self.buffer = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.materials = []

    def add_view(self, data, target=None):
        while len(self.buffer) % 4:
            self.buffer.append(0)
        offset = len(self.buffer)
        self.buffer += data
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_accessor(self, view, atype, ctype, count, normalized=None, minmax=None):
        component_sizes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
        type_components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
        acc = {
            "bufferView": view,
            "componentType": ctype,
            "count": count,
            "type": atype,
        }
        if normalized:
            acc["normalized"] = True
        if minmax is not None:
            acc["min"] = minmax[0]
            acc["max"] = minmax[1]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def add_material(self, name, base_color):
        self.materials.append({"name": name, "pbrMetallicRoughness": {"baseColorFactor": base_color}})
        return len(self.materials) - 1

    def add_points_primitive(self, position_acc, indices_acc=None, attributes=None, material=None):
        prim = {"attributes": {"POSITION": position_acc}, "mode": 0}
        if indices_acc is not None:
            prim["indices"] = indices_acc
        for name, acc in (attributes or {}).items():
            prim["attributes"][name] = acc
        if material is not None:
            prim["material"] = material
        return prim

    def add_mesh(self, name, primitives):
        self.meshes.append({"name": name, "primitives": primitives})
        return len(self.meshes) - 1

    def add_node(self, mesh, name=None):
        node = {"mesh": mesh}
        if name:
            node["name"] = name
        self.nodes.append(node)
        return len(self.nodes) - 1

    def build(self):
        gltf = {
            "asset": {"version": "2.0", "generator": "indexed-points-testgen"},
            "scene": 0,
            "scenes": [{"nodes": list(range(len(self.nodes)))}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.buffer)}],
        }
        if self.materials:
            gltf["materials"] = self.materials

        while len(self.buffer) % 4:
            self.buffer.append(0)
        js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        while len(js) % 4:
            js += b" "
        total = 12 + 8 + len(js) + 8 + len(self.buffer)
        out = struct.pack("<III", 0x46546C67, 2, total)
        out += struct.pack("<I", len(js)) + b"JSON" + js
        out += struct.pack("<I", len(self.buffer)) + b"BIN\x00" + bytes(self.buffer)
        return out


def f32_view(values):
    return struct.pack("<%df" % len(values), *values)


def u16_view(values):
    return struct.pack("<%dH" % len(values), *values)


def u32_view(values):
    return struct.pack("<%dI" % len(values), *values)


def u8_view(values):
    return struct.pack("<%dB" % len(values), *values)


def position_accessor(b, positions):
    flat = [v for p in positions for v in p]
    view = b.add_view(f32_view(flat), target=34962)
    mins = [min(p[k] for p in positions) for k in range(3)]
    maxs = [max(p[k] for p in positions) for k in range(3)]
    return b.add_accessor(view, "VEC3", 5126, len(positions), minmax=(mins, maxs))


def index_accessor(b, indices):
    view = b.add_view(u16_view(indices), target=34963)
    return b.add_accessor(view, "SCALAR", 5123, len(indices))


def normal_accessor(b, normals):
    flat = [v for n in normals for v in n]
    view = b.add_view(f32_view(flat), target=34962)
    return b.add_accessor(view, "VEC3", 5126, len(normals))


def color_accessor(b, colors):
    flat = [v for c in colors for v in c]
    view = b.add_view(u8_view(flat), target=34962)
    return b.add_accessor(view, "VEC4", 5121, len(colors), normalized=True)


def uv_accessor(b, uvs):
    flat = [v for uv in uvs for v in uv]
    view = b.add_view(f32_view(flat), target=34962)
    return b.add_accessor(view, "VEC2", 5126, len(uvs))


def sample_points(n):
    # deterministic pseudo-random point cloud in a non-cubic box so axis/bbox mixups show up
    positions, normals, colors, uvs = [], [], [], []
    seed = 1234567
    for i in range(n):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        x = (seed % 1000) / 1000.0 * 10.0
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        y = (seed % 1000) / 1000.0 * 4.0
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        z = (seed % 1000) / 1000.0 * 6.0 - 2.0
        positions.append((x, y, z))
        normals.append((0.0, 0.0, 1.0 if (i % 2) else -1.0))
        colors.append((i % 256, (i * 7) % 256, (i * 13) % 256, 255))
        uvs.append((x / 10.0, y / 4.0))
    return positions, normals, colors, uvs


def write(path, data):
    with open(path, "wb") as f:
        f.write(data)
    print("wrote", path, len(data), "bytes")


def gen_indexed_points(outdir):
    """Indexed points with position/normal/color/uv and a material."""
    b = GlbBuilder()
    positions, normals, colors, uvs = sample_points(64)
    # shuffled indices with duplicates: exercises remapping while keeping point count
    indices = [(i * 37 + 5) % len(positions) for i in range(48)]
    mat = b.add_material("point-mat", [0.8, 0.2, 0.1, 1.0])
    p = position_accessor(b, positions)
    prim = b.add_points_primitive(
        p,
        index_accessor(b, indices),
        {"NORMAL": normal_accessor(b, normals), "COLOR_0": color_accessor(b, colors), "TEXCOORD_0": uv_accessor(b, uvs)},
        mat,
    )
    mesh = b.add_mesh("indexed-points", [prim])
    b.add_node(mesh, "points-node")
    write(os.path.join(outdir, "indexed_points.glb"), b.build())


def gen_nonindexed_points(outdir):
    """Non-indexed points, regression coverage for the existing path."""
    b = GlbBuilder()
    positions, normals, colors, uvs = sample_points(32)
    p = position_accessor(b, positions)
    prim = b.add_points_primitive(p, None, {"COLOR_0": color_accessor(b, colors)}, None)
    mesh = b.add_mesh("plain-points", [prim])
    b.add_node(mesh)
    write(os.path.join(outdir, "nonindexed_points.glb"), b.build())


def gen_mixed_primitives(outdir):
    """One mesh with a triangle primitive and an indexed points primitive."""
    b = GlbBuilder()
    tri_positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    tri_view = b.add_view(f32_view([v for p in tri_positions for v in p]), target=34962)
    tri_p = b.add_accessor(tri_view, "VEC3", 5126, 3, minmax=([0, 0, 0], [1, 1, 0]))
    tri_i = index_accessor(b, [0, 1, 2])
    tri_prim = {"attributes": {"POSITION": tri_p}, "indices": tri_i, "mode": 4}

    positions, normals, colors, uvs = sample_points(24)
    indices = [(i * 11 + 3) % len(positions) for i in range(20)]
    p = position_accessor(b, positions)
    point_prim = b.add_points_primitive(p, index_accessor(b, indices), {"NORMAL": normal_accessor(b, normals)}, None)

    mesh = b.add_mesh("mixed", [tri_prim, point_prim])
    b.add_node(mesh, "mixed-node")
    write(os.path.join(outdir, "mixed_primitives.glb"), b.build())


def gen_bad_indices(outdir):
    """Two index edge cases:
    - bad_indices.glb: an index references a vertex past the position accessor.
      cgltf_validate rejects this structurally before gltfpack parsing.
    - duplicate_indices.glb: valid range, but indices repeat vertices; this must
      roundtrip safely while preserving the exact point sequence/count."""
    b = GlbBuilder()
    positions, _, _, _ = sample_points(8)
    indices = [0, 1, 2, 7, 99, 3]  # 99 is out of range
    p = position_accessor(b, positions)
    prim = b.add_points_primitive(p, index_accessor(b, indices))
    mesh = b.add_mesh("bad-indices", [prim])
    b.add_node(mesh)
    write(os.path.join(outdir, "bad_indices.glb"), b.build())

    b2 = GlbBuilder()
    positions2, normals2, colors2, _ = sample_points(8)
    indices2 = [0, 1, 2, 7, 7, 3, 3, 0]  # all in range, duplicates present
    p2 = position_accessor(b2, positions2)
    prim2 = b2.add_points_primitive(
        p2, index_accessor(b2, indices2),
        {"NORMAL": normal_accessor(b2, normals2), "COLOR_0": color_accessor(b2, colors2)},
    )
    mesh2 = b2.add_mesh("duplicate-indices", [prim2])
    b2.add_node(mesh2)
    write(os.path.join(outdir, "duplicate_indices.glb"), b2.build())


def gen_attr_mismatch(outdir):
    """Points where COLOR_0 has fewer elements than POSITION."""
    b = GlbBuilder()
    positions, _, colors, _ = sample_points(16)
    p = position_accessor(b, positions)
    bad_colors = colors[:8]
    c = color_accessor(b, bad_colors)
    prim = b.add_points_primitive(p, index_accessor(b, list(range(16))), {"COLOR_0": c})
    mesh = b.add_mesh("attr-mismatch", [prim])
    b.add_node(mesh)
    write(os.path.join(outdir, "attr_mismatch.glb"), b.build())


def gen_empty_points(outdir):
    """Mesh with an empty indexed-points primitive alongside a valid triangle primitive."""
    b = GlbBuilder()
    tri_positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    tri_view = b.add_view(f32_view([v for p in tri_positions for v in p]), target=34962)
    tri_p = b.add_accessor(tri_view, "VEC3", 5126, 3, minmax=([0, 0, 0], [1, 1, 0]))
    tri_prim = {"attributes": {"POSITION": tri_p}, "indices": index_accessor(b, [0, 1, 2]), "mode": 4}

    # empty position accessor (count 0) plus an empty index accessor
    empty_view = b.add_view(b"", target=34962)
    empty_p = b.add_accessor(empty_view, "VEC3", 5126, 0, minmax=([0, 0, 0], [0, 0, 0]))
    empty_i_view = b.add_view(b"", target=34963)
    empty_i = b.add_accessor(empty_i_view, "SCALAR", 5123, 0)
    empty_prim = b.add_points_primitive(empty_p, empty_i)

    mesh = b.add_mesh("empty-points", [tri_prim, empty_prim])
    b.add_node(mesh)
    write(os.path.join(outdir, "empty_points.glb"), b.build())


def gen_large_indexed_points(outdir):
    """Larger indexed point cloud (70k vertices) to exercise 32-bit indices."""
    b = GlbBuilder()
    n = 70000
    positions = [(float(i % 1000), float(i // 1000), float((i * 3) % 97)) for i in range(n)]
    indices = [(i * 104729 + 17) % n for i in range(n // 2)]
    p = position_accessor(b, positions)
    view = b.add_view(u32_view(indices), target=34963)
    i_acc = b.add_accessor(view, "SCALAR", 5125, len(indices))
    prim = b.add_points_primitive(p, i_acc)
    mesh = b.add_mesh("large-points", [prim])
    b.add_node(mesh)
    write(os.path.join(outdir, "large_indexed_points.glb"), b.build())


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    gen_indexed_points(outdir)
    gen_nonindexed_points(outdir)
    gen_mixed_primitives(outdir)
    gen_bad_indices(outdir)
    gen_attr_mismatch(outdir)
    gen_empty_points(outdir)
    gen_large_indexed_points(outdir)


if __name__ == "__main__":
    main()
