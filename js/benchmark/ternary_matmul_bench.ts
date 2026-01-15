/**
 * Ternary Matmul Benchmark
 *
 * Compares three implementations:
 * 1. Baseline: Current implementation with branches
 * 2. Branch-free: Uses (code & 1) - (code >> 1) for weight decode
 * 3. Separate accumulators: No multiplication, pure add/sub (like CUDA kernel)
 */

// ============================================================================
// Implementation 1: Baseline (current)
// ============================================================================
function ternaryMatmulBaseline(
  input: Float32Array,
  weightsPacked: Uint8Array,
  scales: Float32Array,
  inFeatures: number,
  outFeatures: number
): Float32Array {
  const output = new Float32Array(outFeatures);
  const inBytes = Math.ceil(inFeatures / 4);

  for (let n = 0; n < outFeatures; n++) {
    let acc = 0;
    const weightOffset = n * inBytes;

    for (let kb = 0; kb < inBytes; kb++) {
      const packed = weightsPacked[weightOffset + kb];

      for (let i = 0; i < 4 && kb * 4 + i < inFeatures; i++) {
        const code = (packed >> (i * 2)) & 0x3;
        // Branching decode
        const w = code === 1 ? 1 : code === 2 ? -1 : 0;
        const k = kb * 4 + i;
        acc += input[k] * w;
      }
    }

    output[n] = acc * scales[n];
  }

  return output;
}

// ============================================================================
// Implementation 2: Branch-free decode
// ============================================================================
function ternaryMatmulBranchFree(
  input: Float32Array,
  weightsPacked: Uint8Array,
  scales: Float32Array,
  inFeatures: number,
  outFeatures: number
): Float32Array {
  const output = new Float32Array(outFeatures);
  const inBytes = Math.ceil(inFeatures / 4);

  for (let n = 0; n < outFeatures; n++) {
    let acc = 0;
    const weightOffset = n * inBytes;

    for (let kb = 0; kb < inBytes; kb++) {
      const packed = weightsPacked[weightOffset + kb];

      for (let i = 0; i < 4 && kb * 4 + i < inFeatures; i++) {
        const code = (packed >> (i * 2)) & 0x3;
        // Branch-free decode: 00->0, 01->+1, 10->-1
        const w = (code & 1) - (code >> 1);
        const k = kb * 4 + i;
        acc += input[k] * w;
      }
    }

    output[n] = acc * scales[n];
  }

  return output;
}

// ============================================================================
// Implementation 3: Separate accumulators (no multiplication)
// ============================================================================
function ternaryMatmulSeparateAcc(
  input: Float32Array,
  weightsPacked: Uint8Array,
  scales: Float32Array,
  inFeatures: number,
  outFeatures: number
): Float32Array {
  const output = new Float32Array(outFeatures);
  const inBytes = Math.ceil(inFeatures / 4);

  for (let n = 0; n < outFeatures; n++) {
    let accPos = 0;
    let accNeg = 0;
    const weightOffset = n * inBytes;

    for (let kb = 0; kb < inBytes; kb++) {
      const packed = weightsPacked[weightOffset + kb];

      for (let i = 0; i < 4 && kb * 4 + i < inFeatures; i++) {
        const code = (packed >> (i * 2)) & 0x3;
        const k = kb * 4 + i;
        const val = input[k];

        // No multiplication - just accumulate based on code
        // code=1 means +1, code=2 means -1, code=0 means 0
        if (code === 1) {
          accPos += val;
        } else if (code === 2) {
          accNeg += val;
        }
      }
    }

    output[n] = (accPos - accNeg) * scales[n];
  }

  return output;
}

// ============================================================================
// Implementation 4: Separate accumulators + unrolled inner loop
// ============================================================================
function ternaryMatmulUnrolled(
  input: Float32Array,
  weightsPacked: Uint8Array,
  scales: Float32Array,
  inFeatures: number,
  outFeatures: number
): Float32Array {
  const output = new Float32Array(outFeatures);
  const inBytes = Math.ceil(inFeatures / 4);
  const fullBytes = Math.floor(inFeatures / 4);

  for (let n = 0; n < outFeatures; n++) {
    let accPos = 0;
    let accNeg = 0;
    const weightOffset = n * inBytes;

    // Process full bytes (4 values each) - unrolled
    for (let kb = 0; kb < fullBytes; kb++) {
      const packed = weightsPacked[weightOffset + kb];
      const k = kb * 4;

      // Unroll all 4 values
      const code0 = packed & 0x3;
      const code1 = (packed >> 2) & 0x3;
      const code2 = (packed >> 4) & 0x3;
      const code3 = (packed >> 6) & 0x3;

      if (code0 === 1) accPos += input[k];
      else if (code0 === 2) accNeg += input[k];

      if (code1 === 1) accPos += input[k + 1];
      else if (code1 === 2) accNeg += input[k + 1];

      if (code2 === 1) accPos += input[k + 2];
      else if (code2 === 2) accNeg += input[k + 2];

      if (code3 === 1) accPos += input[k + 3];
      else if (code3 === 2) accNeg += input[k + 3];
    }

    // Handle remaining elements
    if (fullBytes < inBytes) {
      const packed = weightsPacked[weightOffset + fullBytes];
      const remaining = inFeatures - fullBytes * 4;
      const k = fullBytes * 4;

      for (let i = 0; i < remaining; i++) {
        const code = (packed >> (i * 2)) & 0x3;
        if (code === 1) accPos += input[k + i];
        else if (code === 2) accNeg += input[k + i];
      }
    }

    output[n] = (accPos - accNeg) * scales[n];
  }

  return output;
}

// ============================================================================
// Implementation 5: Branch-free with bitmasks (no conditionals at all)
// ============================================================================
function ternaryMatmulBitmask(
  input: Float32Array,
  weightsPacked: Uint8Array,
  scales: Float32Array,
  inFeatures: number,
  outFeatures: number
): Float32Array {
  const output = new Float32Array(outFeatures);
  const inBytes = Math.ceil(inFeatures / 4);
  const fullBytes = Math.floor(inFeatures / 4);

  for (let n = 0; n < outFeatures; n++) {
    let acc = 0;
    const weightOffset = n * inBytes;

    // Process full bytes - unrolled with branch-free arithmetic
    for (let kb = 0; kb < fullBytes; kb++) {
      const packed = weightsPacked[weightOffset + kb];
      const k = kb * 4;

      // Extract codes
      const c0 = packed & 0x3;
      const c1 = (packed >> 2) & 0x3;
      const c2 = (packed >> 4) & 0x3;
      const c3 = (packed >> 6) & 0x3;

      // Branch-free: w = (code & 1) - (code >> 1)
      acc += input[k] * ((c0 & 1) - (c0 >> 1));
      acc += input[k + 1] * ((c1 & 1) - (c1 >> 1));
      acc += input[k + 2] * ((c2 & 1) - (c2 >> 1));
      acc += input[k + 3] * ((c3 & 1) - (c3 >> 1));
    }

    // Handle remaining
    if (fullBytes < inBytes) {
      const packed = weightsPacked[weightOffset + fullBytes];
      const remaining = inFeatures - fullBytes * 4;
      const k = fullBytes * 4;

      for (let i = 0; i < remaining; i++) {
        const code = (packed >> (i * 2)) & 0x3;
        acc += input[k + i] * ((code & 1) - (code >> 1));
      }
    }

    output[n] = acc * scales[n];
  }

  return output;
}

// ============================================================================
// Benchmark utilities
// ============================================================================

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
    // Random ternary values: each 2-bit pair is 0, 1, or 2
    let byte = 0;
    for (let j = 0; j < 4; j++) {
      const val = Math.floor(Math.random() * 3); // 0, 1, or 2
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

function verifyCorrectness(
  baseline: Float32Array,
  test: Float32Array,
  name: string,
  tolerance = 1e-5
): boolean {
  if (baseline.length !== test.length) {
    console.error(`${name}: Length mismatch ${baseline.length} vs ${test.length}`);
    return false;
  }
  for (let i = 0; i < baseline.length; i++) {
    const diff = Math.abs(baseline[i] - test[i]);
    if (diff > tolerance) {
      console.error(`${name}: Mismatch at ${i}: ${baseline[i]} vs ${test[i]} (diff=${diff})`);
      return false;
    }
  }
  return true;
}

type MatmulFn = (
  input: Float32Array,
  weightsPacked: Uint8Array,
  scales: Float32Array,
  inFeatures: number,
  outFeatures: number
) => Float32Array;

function benchmark(
  name: string,
  fn: MatmulFn,
  input: Float32Array,
  weights: Uint8Array,
  scales: Float32Array,
  inFeatures: number,
  outFeatures: number,
  warmupIters: number,
  benchIters: number
): { mean: number; std: number; min: number; max: number } {
  // Warmup
  for (let i = 0; i < warmupIters; i++) {
    fn(input, weights, scales, inFeatures, outFeatures);
  }

  // Benchmark
  const times: number[] = [];
  for (let i = 0; i < benchIters; i++) {
    const start = performance.now();
    fn(input, weights, scales, inFeatures, outFeatures);
    const end = performance.now();
    times.push(end - start);
  }

  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const variance = times.reduce((a, b) => a + (b - mean) ** 2, 0) / times.length;
  const std = Math.sqrt(variance);
  const min = Math.min(...times);
  const max = Math.max(...times);

  return { mean, std, min, max };
}

// ============================================================================
// Main benchmark
// ============================================================================

async function main() {
  console.log("=".repeat(70));
  console.log("Ternary Matmul Benchmark");
  console.log("=".repeat(70));
  console.log();

  // Test configurations (typical transformer layer sizes)
  const configs = [
    { inFeatures: 512, outFeatures: 512, name: "Small (512x512)" },
    { inFeatures: 512, outFeatures: 1536, name: "MLP Gate (512x1536)" },
    { inFeatures: 1536, outFeatures: 512, name: "MLP Down (1536x512)" },
    { inFeatures: 512, outFeatures: 2048, name: "Large (512x2048)" },
  ];

  const implementations: [string, MatmulFn][] = [
    ["Baseline (branches)", ternaryMatmulBaseline],
    ["Branch-free decode", ternaryMatmulBranchFree],
    ["Separate accumulators", ternaryMatmulSeparateAcc],
    ["Unrolled + separate acc", ternaryMatmulUnrolled],
    ["Bitmask unrolled", ternaryMatmulBitmask],
  ];

  const warmupIters = 100;
  const benchIters = 500;

  for (const config of configs) {
    console.log(`\n${"─".repeat(70)}`);
    console.log(`Config: ${config.name}`);
    console.log(`  inFeatures: ${config.inFeatures}, outFeatures: ${config.outFeatures}`);
    console.log(`  Warmup: ${warmupIters}, Bench: ${benchIters} iterations`);
    console.log(`${"─".repeat(70)}`);

    // Create test data
    const input = createRandomInput(config.inFeatures);
    const weights = createRandomPackedWeights(config.inFeatures, config.outFeatures);
    const scales = createRandomScales(config.outFeatures);

    // Get baseline result for correctness check
    const baselineResult = ternaryMatmulBaseline(
      input, weights, scales, config.inFeatures, config.outFeatures
    );

    const results: { name: string; mean: number; speedup: number }[] = [];
    let baselineMean = 0;

    for (const [name, fn] of implementations) {
      // Verify correctness
      const testResult = fn(input, weights, scales, config.inFeatures, config.outFeatures);
      const correct = verifyCorrectness(baselineResult, testResult, name);

      if (!correct) {
        console.log(`  ${name}: FAILED correctness check!`);
        continue;
      }

      // Benchmark
      const stats = benchmark(
        name, fn, input, weights, scales,
        config.inFeatures, config.outFeatures,
        warmupIters, benchIters
      );

      if (name === "Baseline (branches)") {
        baselineMean = stats.mean;
      }

      const speedup = baselineMean / stats.mean;
      results.push({ name, mean: stats.mean, speedup });

      console.log(`  ${name.padEnd(25)} ${stats.mean.toFixed(4)}ms ` +
        `(±${stats.std.toFixed(4)}) [${stats.min.toFixed(4)}-${stats.max.toFixed(4)}] ` +
        `${speedup.toFixed(2)}x`);
    }

    // Summary
    const best = results.reduce((a, b) => a.mean < b.mean ? a : b);
    console.log(`\n  Best: ${best.name} (${best.speedup.toFixed(2)}x faster than baseline)`);
  }

  console.log("\n" + "=".repeat(70));
  console.log("Benchmark complete!");
  console.log("=".repeat(70));
}

main().catch(console.error);
