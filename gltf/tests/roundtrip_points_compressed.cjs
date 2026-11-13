// Compressed roundtrip verification using the meshoptimizer JS decoder.
// Decodes EXT/KHR_meshopt_compression bufferViews of indexed POINTS output and
// checks per-point attributes against the source asset. Run with node after roundtrip_points.py.
const fs = require("fs");
const path = require("path");
const MeshoptDecoder = require(path.join(__dirname, "..", "..", "js", "meshopt_decoder.cjs"));

const MODE_DECODER = {
	ATTRIBUTES: "ATTRIBUTES",
	TRIANGLES: "TRIANGLES",
	INDICES: "INDICES",
};
const FILTER = { NONE: "NONE", OCTAHEDRAL: "OCTAHEDRAL", QUATERNION: "QUATERNION", EXPONENTIAL: "EXPONENTIAL", COLOR: "COLOR" };

function readGlb(file) {
	const data = fs.readFileSync(file);
	const jsonLen = data.readUInt32LE(12);
	const doc = JSON.parse(data.slice(20, 20 + jsonLen).toString("utf8"));
	const binOff = 20 + jsonLen + 8;
	const bin = new Uint8Array(data.slice(binOff));
	return { doc, bin };
}

function extOf(view) {
	return view.extensions && (view.extensions.EXT_meshopt_compression || view.extensions.KHR_meshopt_compression);
}

// Decode a bufferView (compressed or plain) into raw bytes matching view.byteLength.
function decodeView(doc, bin, viewIndex) {
	const view = doc.bufferViews[viewIndex];
	const ext = extOf(view);
	if (!ext) return bin.slice(view.byteOffset || 0, (view.byteOffset || 0) + view.byteLength);

	const src = bin.slice(ext.byteOffset, ext.byteOffset + ext.byteLength);
	const target = new Uint8Array(ext.count * ext.byteStride);
	MeshoptDecoder.decodeGltfBuffer(target, ext.count, ext.byteStride, src, MODE_DECODER[ext.mode], FILTER[ext.filter || "NONE"]);
	return target;
}

const CT_SIZE = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const TYPE_COMPS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 };

function decodeAccessor(container, doc, index) {
	const acc = doc.accessors[index];
	const view = doc.bufferViews[acc.bufferView];
	const csize = CT_SIZE[acc.componentType];
	const comps = TYPE_COMPS[acc.type];
	const stride = view.byteStride || csize * comps;
	const bytes = decodeView(doc, container.bin, acc.bufferView);
	const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const viewOffset = acc.byteOffset || 0;
	const rows = [];
	for (let i = 0; i < acc.count; ++i) {
		const off = viewOffset + i * stride;
		const row = [];
		for (let k = 0; k < comps; ++k) {
			let v;
			switch (acc.componentType) {
				case 5120:
					v = dv.getInt8(off + k);
					break;
				case 5121:
					v = dv.getUint8(off + k);
					break;
				case 5122:
					v = dv.getInt16(off + k * 2, true);
					break;
				case 5123:
					v = dv.getUint16(off + k * 2, true);
					break;
				case 5125:
					v = dv.getUint32(off + k * 4, true);
					break;
				default:
					v = dv.getFloat32(off + k * 4, true);
			}
			if (acc.normalized) {
				if (acc.componentType === 5121) v /= 255;
				else if (acc.componentType === 5123) v /= 65535;
				else if (acc.componentType === 5120) v = Math.max(v / 127, -1);
				else if (acc.componentType === 5122) v = Math.max(v / 32767, -1);
			}
			row.push(v);
		}
		rows.push(row);
	}
	return rows;
}

function nodeForMesh(doc, meshIndex) {
	return doc.nodes.find((n) => n.mesh === meshIndex) || {};
}

function dequantizePositions(doc, meshIndex, rows) {
	const node = nodeForMesh(doc, meshIndex);
	const t = node.translation || [0, 0, 0];
	const s = node.scale || [1, 1, 1];
	return rows.map((r) => [t[0] + s[0] * r[0], t[1] + s[1] * r[1], t[2] + s[2] * r[2]]);
}

let failures = 0;
function check(cond, msg) {
	if (cond) console.log("  PASS:", msg);
	else {
		++failures;
		console.log("  FAIL:", msg);
	}
}

function run(srcFile, outFile, label, posTol) {
	console.log(`\n== compressed decode roundtrip [${label}] ==`);
	const src = readGlb(srcFile);
	const out = readGlb(outFile);

	// every output index/vertex view must carry a meshopt compression extension
	const outPrim = out.doc.meshes[0].primitives[0];
	const views = [outPrim.indices, ...Object.values(outPrim.attributes)].map((a) => out.doc.bufferViews[out.doc.accessors[a].bufferView]);
	check(views.every((v) => extOf(v)), "all point bufferViews use meshopt compression");
	check(outPrim.mode === 0, "output mode is POINTS");
	check(outPrim.indices !== undefined, "output stays indexed");

	const srcPrim = src.doc.meshes[0].primitives[0];
	const srcIdx = decodeAccessor(src, src.doc, srcPrim.indices);
	const outIdx = decodeAccessor(out, out.doc, outPrim.indices);
	check(srcIdx.length === outIdx.length, `point count preserved after decode (${outIdx.length})`);

	const srcPos = dequantizePositions(src.doc, 0, decodeAccessor(src, src.doc, srcPrim.attributes.POSITION));
	const outPos = dequantizePositions(out.doc, 0, decodeAccessor(out, out.doc, outPrim.attributes.POSITION));

	const srcPts = srcIdx.map((i) => srcPos[i]);
	const outPts = outIdx.map((i) => outPos[i]);

	// multiset comparison with tolerance
	let mismatches = 0;
	const used = new Array(srcPts.length).fill(false);
	for (const p of outPts) {
		let best = Infinity;
		let bestj = -1;
		for (let j = 0; j < srcPts.length; ++j) {
			if (used[j]) continue;
			const d = Math.max(...srcPts[j].map((v, k) => Math.abs(v - p[k])));
			if (d < best) {
				best = d;
				bestj = j;
			}
		}
		if (best > posTol) ++mismatches;
		else used[bestj] = true;
	}
	check(mismatches === 0, `all decoded positions within tolerance ${posTol.toFixed(5)} (${mismatches} mismatches)`);

	// bounds
	for (let k = 0; k < 3; ++k) {
		const smin = Math.min(...srcPts.map((p) => p[k]));
		const smax = Math.max(...srcPts.map((p) => p[k]));
		const omin = Math.min(...outPts.map((p) => p[k]));
		const omax = Math.max(...outPts.map((p) => p[k]));
		check(Math.abs(smin - omin) <= posTol && Math.abs(smax - omax) <= posTol, `bounds axis ${k} within tolerance`);
	}

	// colors
	{
		const srcCol = decodeAccessor(src, src.doc, srcPrim.attributes.COLOR_0);
		const outCol = decodeAccessor(out, out.doc, outPrim.attributes.COLOR_0);
		const colTol = 2 / 255;
		let bad = 0;
		const usedC = new Array(srcIdx.length).fill(false);
		for (const i of outIdx) {
			let best = Infinity;
			let bestj = -1;
			srcIdx.forEach((si, j) => {
				if (usedC[j]) return;
				const d = Math.max(...srcCol[si].map((v, k) => Math.abs(v - outCol[i][k])));
				if (d < best) {
					best = d;
					bestj = j;
				}
			});
			if (best > colTol) ++bad;
			else usedC[bestj] = true;
		}
		check(bad === 0, `decoded per-point colors within 8-bit tolerance (${bad} mismatches)`);
	}

	// normals: unit length after oct/snorm decode
	{
		const nrm = decodeAccessor(out, out.doc, outPrim.attributes.NORMAL);
		const bad = nrm.filter((n) => Math.abs(Math.hypot(n[0], n[1], n[2]) - 1) > 0.05).length;
		check(bad === 0, `decoded normals unit length (${bad} bad)`);
		const srcNrm = decodeAccessor(src, src.doc, srcPrim.attributes.NORMAL);
		const srcSign = srcIdx.map((i) => Math.sign(srcNrm[i][2])).sort();
		const outSign = outIdx.map((i) => Math.sign(nrm[i][2])).sort();
		check(JSON.stringify(srcSign) === JSON.stringify(outSign), "decoded normal signs preserved");
	}

	// indices reference valid vertices
	const vertexCount = out.doc.accessors[outPrim.attributes.POSITION].count;
	check(outIdx.every((i) => i < vertexCount), "all decoded indices in range");
}

const dataDir = path.join(__dirname, "data");
const outDir = path.join(__dirname, "out");
const posTol = (10 / (1 << 14)) * 2;

async function main() {
	await MeshoptDecoder.ready;

	run(path.join(dataDir, "indexed_points.glb"), path.join(outDir, "indexed_compressed.glb"), "-c", posTol);
	run(path.join(dataDir, "indexed_points.glb"), path.join(outDir, "indexed_compressed_more.glb"), "-cc", posTol);

	console.log(`\n==== compressed decode: ${failures === 0 ? "ALL PASS" : failures + " FAILED"} ====`);
	process.exit(failures ? 1 : 0);
}

main();
