Here’s a tightened, more ruthless roadmap that matches what you just wrote in prose:

> **Make ternary actually efficient, not just cute.**

I’m collapsing the CPU-only phase and putting the **packed CUDA kernel** front and center.

---

# Track 1 – BitTorch Core (Ternary Engine, v0.2.x)

**v0.1.x** – Done: ternary QAT, int8 CUDA kernel, 185+ tests, MNIST/charLM.
**v0.2.x** – Goal: packed ternary **inference** that truly reduces parameter memory and bandwidth.

Structure v0.2 like this:

* **v0.2.0** – Packed CUDA kernel + `TernaryLinearInference` (critical path)
* **v0.2.1** – Export pipeline + model-level conversion
* **v0.2.2** – Memory/latency benchmarks (prove the wins)
* **v0.2.3** – Robustness + Jetson-oriented polish

No INT4, no extra toys, no backward kernel work in this track.

---

## v0.2.0 – Packed CUDA Kernel + Inference Module

**Goal:** Replace the current “pack → unpack → float matmul” with a **packed CUDA kernel** that reads 2-bit ternary weights directly and never materializes full float/int8 weights.

### 0. Precondition (format) ✅ DONE

Format spec finalized in `docs/ternary_format.md`:

* Encoding: `00 = 0`, `01 = +1`, `10 = -1`, `11 = reserved`
* 4 weights per `uint8` (2 bits each)
* Row-major over `(out_features, in_features)`, padded K to multiple of 4

Also implemented:
* `bittorch/quant/ternary_packed.py` - pack/unpack utilities
* `bittorch/nn/ternary_linear_infer.py` - TernaryLinearInference (CPU path)
* `bittorch/utils/convert.py` - conversion utilities
* `tests/test_packed_ternary.py` - 33 tests
* `examples/inference_demo.py` - demo showing 14x memory reduction

### 1. Packed CUDA kernel ✅ DONE

New file: `csrc/kernels/ternary_gemm_packed.cu`

Implemented:
* Baseline kernel (one thread per output) + tiled kernel (32x32 tiles, shared memory)
* Reads packed uint8 weights directly, unpacks in registers
* Uses add/sub instead of multiply for ternary accumulation
* Supports FP16 and FP32 inputs
* Environment variable `BITTORCH_PACKED_KERNEL=baseline|tiled` for kernel selection

* **Signature** (C++ level):

  ```cpp
  Tensor ternary_linear_packed_forward(
      const Tensor& x,         // [B, K], fp16/fp32
      const Tensor& w_packed,  // packed uint8, [N, K_packed_bytes]
      const Tensor& scale,     // [N], fp16/fp32
      const c10::optional<Tensor>& bias  // [N]
  );
  ```

* **Core behavior:**

  * For each output tile:

    * Load a tile of `x` (as now, using your tiled kernel structure).
    * Load `w_packed` from global memory.
    * **Unpack in registers/shared memory:**

      * 4 weights per byte with bit ops.
      * Map {0,1,2} → {0,+1,-1}.
    * Accumulate in FP32:

      * `w = 0` → skip
      * `w = +1` → `acc += x`
      * `w = -1` → `acc -= x`
    * Apply per-output-channel `scale` and optional `bias`.
  * Never create a full `[N, K]` `w_tern` in global memory.

* **Constraints:**

  * Reuse your existing tile sizes / shared mem scheme where possible.
  * Keep the old int8 kernel as a **debug / baseline** path.

### 2. Python bindings & ops wrapper ✅ DONE

Implemented:
* `csrc/core/dispatch.cpp` - `ternary_linear_packed_forward` dispatch with validation
* `csrc/bindings/bittorch_bindings.cpp` - Python bindings
* `bittorch/ops/ternary_linear_packed.py` - Python wrapper with `has_packed_cuda_support()`

### 3. Wire into `TernaryLinearInference` ✅ DONE

File: `bittorch/nn/ternary_linear_infer.py`

Implemented:
* GPU path: calls `ternary_linear_packed_forward` directly
* CPU fallback: unpack + matmul (slow but simple)
* All buffers, no Parameters, `requires_grad=False`

### 4. Tests ✅ DONE

File: `tests/test_packed_kernel.py` - 36 tests covering:
* Basic correctness vs unpacked reference
* Various shapes (4x8 to 3072x768 transformer-like)
* K not divisible by 4 (padding path)
* Batch sizes 1, 2, 4, 8, 16, 32, 64
* FP16 and FP32 dtypes
* Edge cases: all zeros, all +1, all -1
* TernaryLinearInference integration on CUDA
* Determinism and reproducibility

### v0.2.0 Exit Criteria ✅ MET

* Packed CUDA kernel runs and matches float/unpacked reference within tolerance.
* `TernaryLinearInference` uses packed CUDA kernel on GPU and no FP master weights.
* Old int8 kernel exists only as baseline/debug, not used in inference module by default.
* 201 total tests passing (36 new for packed kernel)

---

## v0.2.1 – Export Pipeline: Training → Packed Inference

**Goal:** One-step conversion from “trained model with `TernaryLinear`” to “deployment model with `TernaryLinearInference` and packed weights”.

### 1. Layer-level export

File: `bittorch/utils/export.py`

* `export_linear_to_inference(module: TernaryLinear) -> TernaryLinearInference`:

  * Take `module.weight` (full-precision master weights).
  * Quantize with your existing ternary quantizer (using frozen config).
  * Compute per-channel `scale`.
  * Pack ternary weights into `uint8` format.
  * Return `TernaryLinearInference` with `weight_packed`, `scale`, `bias`.

### 2. Model-level export

* `export_for_inference(model: nn.Module) -> nn.Module`:

  * Recursively walk modules.
  * Replace `TernaryLinear` with `TernaryLinearInference` via `export_linear_to_inference`.
  * Optionally support a predicate to skip some layers (e.g., output head).

* Optional convenience:

  * `save_ternary_inference_checkpoint(model, path)` – saves only buffers needed for inference.
  * `load_ternary_inference_checkpoint(path) -> nn.Module`.

### 3. Tests

File: `tests/test_export_inference.py`

* Build a small MLP with `TernaryLinear`, random weights.
* Export to inference model.
* For random `x`:

  * Run training model with quantization frozen.
  * Run inference model.
  * Compare outputs.

### v0.2.1 Exit Criteria

* Single call (`export_for_inference`) converts a training model to inference version.
* Output of exported model matches training-time quantized forward (within tolerance).
* No training-only state (optimizer, master weights) is required for inference.

---

## v0.2.2 – Memory & Latency Benchmarks

**Goal:** Prove the story with numbers: parameter memory and runtime behavior.

### 1. Benchmarks

File: `benchmark/bench_packed_inference.py`

* Compare three variants:

  1. `nn.Linear` FP16/FP32.
  2. `TernaryLinear` (training module with FP master weights).
  3. `TernaryLinearInference` (packed).

* Shapes:

  * MLP-ish: `(B=32, K=1024, N=4096)`
  * Transformer-ish: `(B=32, seq=512, hidden=768)` → treat as `x.view(-, hidden)` for dense.
  * Small-batch: `(B=1, K=hidden, N=hidden)`.

* For each:

  * Measure `torch.cuda.max_memory_allocated()` after warmup.
  * Measure average forward time over N iterations.

### 2. Documentation

* README / docs:

  * “How to export a trained BitTorch model to packed inference.”

  * Simple code snippet:

    ```python
    model = MyModel(...)
    # train with TernaryLinear ...
    model_inf = export_for_inference(model)
    ```

  * Include a small table:

    | Layer / Model | FP16 params | Ternary packed params | VRAM ratio | Latency ratio |
    | ------------- | ----------: | --------------------: | ---------: | ------------: |

### v0.2.2 Exit Criteria

* Benchmark script runs and shows:

  * Parameter memory clearly reduced (8–16× per weight, plus scales).
  * GPU memory footprint of full models reduced meaningfully.
  * Latency is at least reasonable (even if not yet faster than cuBLAS).

---

## v0.2.3 – Robustness & “Jetson-Ready” Inference

**Goal:** Make inference robust across shapes and small devices.

### 1. Robustness tests

File: `tests/test_infer_robustness.py`

* Test:

  * Zero inputs.
  * Very large/small inputs.
  * Various shapes:

    * `B` in {1, 2, 4, 8, 32}.
    * `K, N` not multiples of 4.
  * Deterministic outputs for fixed seeds.

### 2. Jetson constraints

* Validate:

  * Block sizes, shared mem usage are safe for Jetson-class GPUs.
  * Add compute capability guards if necessary.

* Env toggle:

  * `BITTORCH_INFER_KERNEL=packed|baseline` to fall back to a simpler kernel for debugging if needed.

### 3. Deployment guide

File: `docs/deployment_guide.md`

* Steps:

  * Install BitTorch on Jetson.
  * Train model with `TernaryLinear` on bigger GPU.
  * `export_for_inference`.
  * Copy to Jetson and run inference.

* Notes:

  * Expected memory savings ranges.
  * Latency caveats.
  * Known limitations.

### v0.2.3 Exit Criteria

* Inference path is stable on your test GPUs, including at least one “small” device.
* Clear docs for exporting and running ternary-packed models.

---

## Final picture

After this roadmap, BitTorch’s **core value prop** becomes:

> * Train with `TernaryLinear` (QAT, FP master weights, existing v0.1.x stack)
> * Export with `export_for_inference()`
> * Deploy with `TernaryLinearInference` using **packed 2-bit weights** and a CUDA kernel that **never unpacks to full floats**.

That’s no longer “a nice ternary research toy”; that’s a focused BitNet-style ternary engine with a real deployment story.
