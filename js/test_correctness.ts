import { ternaryMatmulSingleInto } from "./src/core/TernaryLinear.js";
import type { TernaryLayer } from "./src/core/types.js";

// Original baseline (known working)
function ternaryMatmulBaseline(
  input: Float32Array,
  layer: TernaryLayer,
  output: Float32Array
): void {
  const { inFeatures, outFeatures, weightsPacked, scales } = layer;
  const inBytes = Math.ceil(inFeatures / 4);

  for (let n = 0; n < outFeatures; n++) {
    let acc = 0;
    const weightOffset = n * inBytes;

    for (let kb = 0; kb < inBytes; kb++) {
      const packed = weightsPacked[weightOffset + kb];

      for (let i = 0; i < 4 && kb * 4 + i < inFeatures; i++) {
        const code = (packed >> (i * 2)) & 0x3;
        const w = code === 1 ? 1 : code === 2 ? -1 : 0;
        const k = kb * 4 + i;
        acc += input[k] * w;
      }
    }

    output[n] = acc * scales[n];
  }
}

// Test with known values
const inFeatures = 8;
const outFeatures = 4;
const inBytes = Math.ceil(inFeatures / 4);

// Create simple test input
const input = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);

// Create weights: all +1 for simplicity (code=1 for each position)
// Each byte: 01 01 01 01 = 0x55
const weightsPacked = new Uint8Array(outFeatures * inBytes);
weightsPacked.fill(0x55); // All +1

const scales = new Float32Array([1, 1, 1, 1]);

const layer: TernaryLayer = { inFeatures, outFeatures, weightsPacked, scales };

const outBaseline = new Float32Array(outFeatures);
const outOptimized = new Float32Array(outFeatures);

ternaryMatmulBaseline(input, layer, outBaseline);
ternaryMatmulSingleInto(input, layer, outOptimized);

console.log("Input:", Array.from(input));
console.log("Expected (all +1 weights):", 1+2+3+4+5+6+7+8, "= 36 for each output");
console.log("Baseline:", Array.from(outBaseline));
console.log("Optimized:", Array.from(outOptimized));

// Test with mixed weights
// Let's be explicit about bit layout:
// code 0 = 00, code 1 = 01, code 2 = 10
// Byte packing: bits [1:0]=pos0, [3:2]=pos1, [5:4]=pos2, [7:6]=pos3
// Want weights [0, +1, -1, +1] = codes [0, 1, 2, 1]
// Binary: 01_10_01_00 = 0x64
const mixedWeights = new Uint8Array(outFeatures * inBytes);
mixedWeights[0] = 0x64; // positions 0-3: codes [0, 1, 2, 1] = weights [0, +1, -1, +1]
mixedWeights[1] = 0x64; // positions 4-7: same

const mixedLayer: TernaryLayer = { inFeatures, outFeatures, weightsPacked: mixedWeights, scales };

const outMixedBaseline = new Float32Array(outFeatures);
const outMixedOptimized = new Float32Array(outFeatures);

ternaryMatmulBaseline(input, mixedLayer, outMixedBaseline);
ternaryMatmulSingleInto(input, mixedLayer, outMixedOptimized);

// Manual calculation: weights [0,+1,-1,+1,0,+1,-1,+1]
// 0*1 + 1*2 + (-1)*3 + 1*4 + 0*5 + 1*6 + (-1)*7 + 1*8 = 0+2-3+4+0+6-7+8 = 10
console.log("\nMixed weights [0,+1,-1,+1,0,+1,-1,+1]:");
console.log("Expected: 0*1 + 1*2 + (-1)*3 + 1*4 + 0*5 + 1*6 + (-1)*7 + 1*8 =", 0+2-3+4+0+6-7+8);
console.log("Baseline:", Array.from(outMixedBaseline));
console.log("Optimized:", Array.from(outMixedOptimized));

// Check if they match
const match1 = outBaseline.every((v, i) => Math.abs(v - outOptimized[i]) < 1e-6);
const match2 = outMixedBaseline.every((v, i) => Math.abs(v - outMixedOptimized[i]) < 1e-6);
console.log("\nAll +1 test:", match1 ? "✓ PASS" : "✗ FAIL");
console.log("Mixed test:", match2 ? "✓ PASS" : "✗ FAIL");
