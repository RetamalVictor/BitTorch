import { ternaryMatmul } from "./src/core/TernaryLinear.js";
import type { TernaryLayer } from "./src/core/types.js";

// Original baseline (known working)
function ternaryMatmulBaseline(
  input: Float32Array,
  layer: TernaryLayer,
  seqLen: number
): Float32Array {
  const { inFeatures, outFeatures, weightsPacked, scales } = layer;
  const output = new Float32Array(seqLen * outFeatures);
  const inBytes = Math.ceil(inFeatures / 4);

  for (let t = 0; t < seqLen; t++) {
    const inputOffset = t * inFeatures;
    const outputOffset = t * outFeatures;

    for (let n = 0; n < outFeatures; n++) {
      let acc = 0;
      const weightOffset = n * inBytes;

      for (let kb = 0; kb < inBytes; kb++) {
        const packed = weightsPacked[weightOffset + kb];

        for (let i = 0; i < 4 && kb * 4 + i < inFeatures; i++) {
          const code = (packed >> (i * 2)) & 0x3;
          const w = code === 1 ? 1 : code === 2 ? -1 : 0;
          const k = kb * 4 + i;
          acc += input[inputOffset + k] * w;
        }
      }

      output[outputOffset + n] = acc * scales[n];
    }
  }

  return output;
}

// Test batch version
const inFeatures = 8;
const outFeatures = 4;
const seqLen = 3;

// Input: 3 vectors
const input = new Float32Array([
  1, 2, 3, 4, 5, 6, 7, 8,       // t=0
  10, 20, 30, 40, 50, 60, 70, 80, // t=1
  -1, -2, -3, -4, -5, -6, -7, -8  // t=2
]);

// All +1 weights
const weightsPacked = new Uint8Array((outFeatures * Math.ceil(inFeatures / 4)));
weightsPacked.fill(0x55);

const scales = new Float32Array([1, 1, 1, 1]);

const layer: TernaryLayer = { inFeatures, outFeatures, weightsPacked, scales };

const outBaseline = ternaryMatmulBaseline(input, layer, seqLen);
const outOptimized = ternaryMatmul(input, layer, seqLen);

console.log("Batch test (seqLen=3, all +1 weights):");
console.log("Expected t=0:", 1+2+3+4+5+6+7+8, "= 36");
console.log("Expected t=1:", 10+20+30+40+50+60+70+80, "= 360");
console.log("Expected t=2:", -1-2-3-4-5-6-7-8, "= -36");

console.log("\nBaseline output:");
for (let t = 0; t < seqLen; t++) {
  console.log(`  t=${t}:`, Array.from(outBaseline.slice(t * outFeatures, (t + 1) * outFeatures)));
}

console.log("\nOptimized output:");
for (let t = 0; t < seqLen; t++) {
  console.log(`  t=${t}:`, Array.from(outOptimized.slice(t * outFeatures, (t + 1) * outFeatures)));
}

const match = outBaseline.every((v, i) => Math.abs(v - outOptimized[i]) < 1e-6);
console.log("\nBatch test:", match ? "✓ PASS" : "✗ FAIL");

if (!match) {
  console.log("\nDifferences:");
  for (let i = 0; i < outBaseline.length; i++) {
    if (Math.abs(outBaseline[i] - outOptimized[i]) > 1e-6) {
      console.log(`  [${i}]: baseline=${outBaseline[i]}, optimized=${outOptimized[i]}`);
    }
  }
}
