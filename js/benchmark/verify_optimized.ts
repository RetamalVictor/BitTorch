/**
 * Verify that the production ternary matmul is using the optimized implementation
 */

import { ternaryMatmulSingleInto } from "../src/core/TernaryLinear.js";
import type { TernaryLayer } from "../src/core/types.js";

function createRandomInput(size: number): Float32Array {
  const arr = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    arr[i] = (Math.random() - 0.5) * 2;
  }
  return arr;
}

function createRandomPackedWeights(inFeatures: number, outFeatures: number): Uint8Array {
  const inBytes = Math.ceil(inFeatures / 4);
  const arr = new Uint8Array(outFeatures * inBytes);
  for (let i = 0; i < arr.length; i++) {
    let byte = 0;
    for (let j = 0; j < 4; j++) {
      const val = Math.floor(Math.random() * 3);
      byte |= val << (j * 2);
    }
    arr[i] = byte;
  }
  return arr;
}

function createRandomScales(outFeatures: number): Float32Array {
  const arr = new Float32Array(outFeatures);
  for (let i = 0; i < outFeatures; i++) {
    arr[i] = Math.random() * 0.1 + 0.01;
  }
  return arr;
}

// Old baseline implementation for comparison
function ternaryMatmulSingleIntoBaseline(
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

async function main() {
  console.log("Verifying production ternaryMatmulSingleInto is optimized...\n");

  const inFeatures = 512;
  const outFeatures = 1536;
  const warmupIters = 100;
  const benchIters = 500;

  const input = createRandomInput(inFeatures);
  const weightsPacked = createRandomPackedWeights(inFeatures, outFeatures);
  const scales = createRandomScales(outFeatures);
  const inBytes = Math.ceil(inFeatures / 4);

  const layer: TernaryLayer = { inFeatures, outFeatures, weightsPacked, scales };
  const outputBaseline = new Float32Array(outFeatures);
  const outputOptimized = new Float32Array(outFeatures);

  // Verify correctness
  ternaryMatmulSingleIntoBaseline(input, layer, outputBaseline);
  ternaryMatmulSingleInto(input, layer, outputOptimized);

  let maxDiff = 0;
  for (let i = 0; i < outFeatures; i++) {
    const diff = Math.abs(outputBaseline[i] - outputOptimized[i]);
    if (diff > maxDiff) maxDiff = diff;
  }
  console.log(`Correctness check: max diff = ${maxDiff.toExponential(2)} (should be ~0)`);

  // Warmup
  for (let i = 0; i < warmupIters; i++) {
    ternaryMatmulSingleInto(input, layer, outputOptimized);
    ternaryMatmulSingleIntoBaseline(input, layer, outputBaseline);
  }

  // Benchmark baseline
  let baselineTime = 0;
  for (let i = 0; i < benchIters; i++) {
    const start = performance.now();
    ternaryMatmulSingleIntoBaseline(input, layer, outputBaseline);
    baselineTime += performance.now() - start;
  }
  baselineTime /= benchIters;

  // Benchmark optimized (production)
  let optimizedTime = 0;
  for (let i = 0; i < benchIters; i++) {
    const start = performance.now();
    ternaryMatmulSingleInto(input, layer, outputOptimized);
    optimizedTime += performance.now() - start;
  }
  optimizedTime /= benchIters;

  const speedup = baselineTime / optimizedTime;

  console.log(`\nBenchmark results (${inFeatures}x${outFeatures}, ${benchIters} iterations):`);
  console.log(`  Baseline (old):    ${baselineTime.toFixed(4)} ms/call`);
  console.log(`  Optimized (prod):  ${optimizedTime.toFixed(4)} ms/call`);
  console.log(`  Speedup:           ${speedup.toFixed(2)}x`);

  if (speedup > 3) {
    console.log(`\n✓ Production code is using the optimized implementation!`);
  } else {
    console.log(`\n✗ WARNING: Production code may not be optimized (expected >3x speedup)`);
  }
}

main().catch(console.error);
