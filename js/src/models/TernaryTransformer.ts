/**
 * TernaryTransformer - LLaMA-style transformer with ternary quantization
 *
 * Main model class for browser-based inference with ternary quantized weights.
 *
 * @module models/TernaryTransformer
 */

import type {
  TransformerConfig,
  TransformerBlock,
  TernaryLayer,
  LayerKVCache,
  GenerateOptions,
  MemoryStats,
} from "../core/types.js";
import { ternaryMatmul, ternaryMatmulSingleInto, matmulFP32, matmulFP32Into } from "../core/TernaryLinear.js";
import { rmsNorm, rmsNormSingleInto } from "../core/RMSNorm.js";
import { embeddingLookup } from "../core/Embedding.js";
import {
  initRoPECache,
  applyRoPE,
  applyRoPESingle,
  applyRoPEToCache,
  applyRoPEToSinglePos,
  type RoPECache,
} from "../core/RoPE.js";
import { SafeTensorsLoader } from "../loaders/SafeTensorsLoader.js";
import { BPETokenizer } from "../tokenizers/BPETokenizer.js";
import { WebGPUBackend } from "../gpu/WebGPUBackend.js";

/**
 * TernaryTransformer - LLaMA-style transformer with ternary quantization.
 *
 * @example
 * ```typescript
 * const model = await TernaryTransformer.load('/path/to/model/');
 *
 * const output = await model.generate('Hello, ', {
 *   maxTokens: 50,
 *   temperature: 0.8,
 *   onToken: (token, stats) => console.log(token)
 * });
 *
 * model.destroy();
 * ```
 */
export class TernaryTransformer {
  private _config: TransformerConfig;
  private tokenizer: BPETokenizer;

  // Model weights
  private embedding: Float32Array;
  private blocks: TransformerBlock[] = [];
  private normWeight: Float32Array;
  private head: Float32Array;

  // RoPE cache
  private ropeCache: RoPECache;

  // KV Cache
  private kvCache: LayerKVCache[] = [];
  private cacheSeqLen: number = 0;

  // GPU acceleration
  private gpu: WebGPUBackend | null = null;

  // Generation control
  private stopRequested: boolean = false;

  // Memory tracking
  private ternaryWeights = 0;
  private fp16Weights = 0;

  // Pre-allocated decode buffers (reused across tokens)
  private decodeBuffers: {
    hidden: Float32Array;
    normed: Float32Array;
    attnOut: Float32Array;
    mlpOut: Float32Array;
    q: Float32Array;
    kv: Float32Array;
    gate: Float32Array;
    up: Float32Array;
    scores: Float32Array;
    logits: Float32Array;
    softmax: Float32Array;
    projTemp: Float32Array;
  } | null = null;

  private constructor(
    config: TransformerConfig,
    tokenizer: BPETokenizer,
    embedding: Float32Array,
    blocks: TransformerBlock[],
    normWeight: Float32Array,
    head: Float32Array,
    ropeCache: RoPECache,
    gpu: WebGPUBackend | null
  ) {
    this._config = config;
    this.tokenizer = tokenizer;
    this.embedding = embedding;
    this.blocks = blocks;
    this.normWeight = normWeight;
    this.head = head;
    this.ropeCache = ropeCache;
    this.gpu = gpu;
  }

  /**
   * Load model from a directory containing config.json, tokenizer.json, and model.safetensors.
   *
   * @param basePath - URL or path to model directory
   * @returns Promise resolving to TernaryTransformer instance
   */
  static async load(basePath: string): Promise<TernaryTransformer> {
    // Load config
    const configResp = await fetch(`${basePath}/config.json`);
    if (!configResp.ok) throw new Error("Failed to load config.json");
    const configJson = await configResp.json();

    const config: TransformerConfig = {
      vocabSize: configJson.vocab_size,
      dim: configJson.dim,
      nLayers: configJson.n_layers,
      nHeads: configJson.n_heads,
      nKvHeads: configJson.n_kv_heads ?? configJson.n_heads,
      maxSeqLen: configJson.max_seq_len,
    };

    // Load tokenizer
    const tokenizer = await BPETokenizer.fromUrl(`${basePath}/tokenizer.json`);

    // Load SafeTensors
    const loader = await SafeTensorsLoader.fromUrl(
      `${basePath}/model.safetensors`
    );

    // Load embedding
    const tokTensor = loader.getTensorFloat32("tok.weight");
    const embedding = tokTensor.data;

    // Load head
    const headTensor = loader.getTensorFloat32("head.weight");
    const head = headTensor.data;

    // Load final norm
    const normTensor = loader.getTensorFloat32("norm.weight");
    const normWeight = normTensor.data;

    // Load transformer blocks
    const blocks: TransformerBlock[] = [];
    let ternaryWeights = 0;
    const fp16Weights = config.vocabSize * config.dim * 2; // embedding + head

    for (let i = 0; i < config.nLayers; i++) {
      const prefix = `blocks.${i}`;

      // RMSNorm weights
      const norm1 = loader.getTensorFloat32(`${prefix}.norm1.weight`);
      const norm2 = loader.getTensorFloat32(`${prefix}.norm2.weight`);

      // Helper to load ternary layer
      const loadTernary = (name: string): TernaryLayer => {
        const weightTensor = loader.getTensorUint8(`${name}.weight_packed`);
        const scaleTensor = loader.getTensorFloat32(`${name}.scale`);
        const outFeatures = weightTensor.shape[0];
        const inFeatures = weightTensor.shape[1] * 4;
        ternaryWeights += outFeatures * inFeatures;
        return {
          weightsPacked: weightTensor.data,
          scales: scaleTensor.data,
          outFeatures,
          inFeatures,
        };
      };

      blocks.push({
        norm1Weight: norm1.data,
        qProj: loadTernary(`${prefix}.attn.q_proj`),
        kvProj: loadTernary(`${prefix}.attn.kv_proj`),
        proj: loadTernary(`${prefix}.attn.proj`),
        norm2Weight: norm2.data,
        wGate: loadTernary(`${prefix}.mlp.w_gate`),
        wUp: loadTernary(`${prefix}.mlp.w_up`),
        wDown: loadTernary(`${prefix}.mlp.w_down`),
      });
    }

    // Initialize RoPE cache
    const headDim = config.dim / config.nHeads;
    const ropeCache = initRoPECache(config.maxSeqLen, headDim);

    // Try to initialize GPU
    let gpu: WebGPUBackend | null = null;
    try {
      gpu = await WebGPUBackend.create();
      if (gpu) {
        // Upload all weights to GPU
        for (let i = 0; i < blocks.length; i++) {
          const block = blocks[i];
          const prefix = `block${i}`;
          gpu.uploadLayer(`${prefix}_qProj`, block.qProj);
          gpu.uploadLayer(`${prefix}_kvProj`, block.kvProj);
          gpu.uploadLayer(`${prefix}_proj`, block.proj);
          gpu.uploadLayer(`${prefix}_wGate`, block.wGate);
          gpu.uploadLayer(`${prefix}_wUp`, block.wUp);
          gpu.uploadLayer(`${prefix}_wDown`, block.wDown);
        }
      }
    } catch {
      gpu = null;
    }

    const model = new TernaryTransformer(
      config,
      tokenizer,
      embedding,
      blocks,
      normWeight,
      head,
      ropeCache,
      gpu
    );
    model.ternaryWeights = ternaryWeights;
    model.fp16Weights = fp16Weights;
    model.initDecodeBuffers();

    return model;
  }

  /**
   * Initialize pre-allocated buffers for decode (called once after load).
   */
  private initDecodeBuffers(): void {
    const { dim, nHeads, nKvHeads, maxSeqLen, vocabSize } = this._config;
    const headDim = dim / nHeads;
    const kvDim = nKvHeads * headDim * 2; // K and V concatenated
    const mlpHiddenDim = this.blocks[0].wGate.outFeatures;

    this.decodeBuffers = {
      hidden: new Float32Array(dim),
      normed: new Float32Array(dim),
      attnOut: new Float32Array(dim),
      mlpOut: new Float32Array(dim),
      q: new Float32Array(dim),
      kv: new Float32Array(kvDim),
      gate: new Float32Array(mlpHiddenDim),
      up: new Float32Array(mlpHiddenDim),
      scores: new Float32Array(maxSeqLen),
      logits: new Float32Array(vocabSize),
      softmax: new Float32Array(vocabSize),
      projTemp: new Float32Array(dim),
    };
  }

  /**
   * Get model configuration.
   */
  get config(): TransformerConfig {
    return { ...this._config };
  }

  /**
   * Check if GPU acceleration is enabled.
   */
  get isGPUEnabled(): boolean {
    return this.gpu !== null;
  }

  /**
   * Get memory usage statistics.
   */
  get memoryStats(): MemoryStats {
    const packedWeightsKB = (this.ternaryWeights * 0.25) / 1024;
    const ternaryAsFP16KB = (this.ternaryWeights * 2) / 1024;
    const actualFP16KB = (this.fp16Weights * 2) / 1024;

    const numScales = this.blocks.reduce(
      (acc, b) =>
        acc +
        b.qProj.outFeatures +
        b.kvProj.outFeatures +
        b.proj.outFeatures +
        b.wGate.outFeatures +
        b.wUp.outFeatures +
        b.wDown.outFeatures,
      0
    );
    const scalesKB = (numScales * 4) / 1024;

    return {
      packedWeightsKB: packedWeightsKB + scalesKB,
      fp16EquivalentKB: ternaryAsFP16KB + actualFP16KB,
      compressionRatio:
        (ternaryAsFP16KB + actualFP16KB) /
        (packedWeightsKB + scalesKB + actualFP16KB),
      scalesKB,
    };
  }

  /**
   * Generate text continuation.
   *
   * @param prompt - Starting text
   * @param options - Generation options
   * @returns Generated text
   */
  async generate(prompt: string, options: GenerateOptions = {}): Promise<string> {
    const { maxTokens = 100, temperature = 0.8, onToken } = options;

    this.stopRequested = false;
    const startTime = performance.now();

    // Tokenize prompt
    const tokens = this.tokenizer.encode(prompt);

    // Initialize KV cache
    this.initKVCache();

    // Prefill: process all prompt tokens
    let logits = await this.forwardPrefill(tokens);
    let generated = "";

    for (let step = 0; step < maxTokens; step++) {
      if (this.stopRequested) break;

      // Sample next token
      const nextToken = this.sample(logits, temperature);

      // Decode
      const decoded = this.tokenizer.decodeToken(nextToken);
      generated += decoded;

      // Report progress
      if (onToken) {
        const elapsedMs = performance.now() - startTime;
        onToken(decoded, {
          tokensPerSecond: ((step + 1) / elapsedMs) * 1000,
          totalTokens: step + 1,
          elapsedMs,
        });
      }

      // Yield to UI
      await new Promise((resolve) => setTimeout(resolve, 0));

      // Check max sequence length
      if (this.cacheSeqLen >= this._config.maxSeqLen - 1) break;

      // Decode step
      logits = await this.forwardDecode(nextToken);
    }

    return generated;
  }

  /**
   * Stop ongoing generation.
   */
  stop(): void {
    this.stopRequested = true;
  }

  /**
   * Release all resources.
   */
  destroy(): void {
    this.gpu?.destroy();
    this.gpu = null;
  }

  // ============================================
  // Private Methods
  // ============================================

  private initKVCache(): void {
    const { nLayers, nKvHeads, maxSeqLen, dim, nHeads } = this._config;
    const headDim = dim / nHeads;

    this.kvCache = [];
    for (let i = 0; i < nLayers; i++) {
      this.kvCache.push({
        k: new Float32Array(nKvHeads * maxSeqLen * headDim),
        v: new Float32Array(nKvHeads * maxSeqLen * headDim),
      });
    }
    this.cacheSeqLen = 0;
  }

  private async forwardPrefill(tokens: number[]): Promise<Float32Array> {
    const { dim, nHeads, nLayers } = this._config;
    const seqLen = tokens.length;
    const headDim = dim / nHeads;

    // Embedding lookup
    let hidden = embeddingLookup(tokens, this.embedding, dim);

    // Process blocks
    for (let i = 0; i < nLayers; i++) {
      const block = this.blocks[i];
      const cache = this.kvCache[i];

      // Pre-attention RMSNorm
      const normed1 = rmsNorm(hidden, block.norm1Weight, seqLen, dim);

      // Attention
      const attnOut = await this.attentionPrefill(
        normed1,
        block,
        cache,
        seqLen,
        headDim,
        i
      );

      // Residual
      for (let j = 0; j < hidden.length; j++) {
        hidden[j] += attnOut[j];
      }

      // Pre-MLP RMSNorm
      const normed2 = rmsNorm(hidden, block.norm2Weight, seqLen, dim);

      // MLP (SwiGLU)
      const mlpOut = await this.swiglu(normed2, block, seqLen, i);

      // Residual
      for (let j = 0; j < hidden.length; j++) {
        hidden[j] += mlpOut[j];
      }
    }

    this.cacheSeqLen = seqLen;

    // Final RMSNorm
    const finalHidden = rmsNorm(hidden, this.normWeight, seqLen, dim);

    // Get last token's hidden state
    const lastHidden = new Float32Array(dim);
    const lastOffset = (seqLen - 1) * dim;
    for (let d = 0; d < dim; d++) {
      lastHidden[d] = finalHidden[lastOffset + d];
    }

    // Output projection
    return matmulFP32(lastHidden, this.head, dim, this._config.vocabSize);
  }

  private async forwardDecode(token: number): Promise<Float32Array> {
    const { dim, nHeads, nLayers, vocabSize } = this._config;
    const headDim = dim / nHeads;
    const pos = this.cacheSeqLen;
    const buf = this.decodeBuffers!;

    // Embedding lookup into hidden buffer
    const srcOffset = token * dim;
    for (let d = 0; d < dim; d++) {
      buf.hidden[d] = this.embedding[srcOffset + d];
    }

    // Process blocks
    for (let i = 0; i < nLayers; i++) {
      const block = this.blocks[i];
      const cache = this.kvCache[i];

      // RMSNorm -> normed buffer
      rmsNormSingleInto(buf.hidden, block.norm1Weight, buf.normed);

      // Attention -> attnOut buffer
      await this.attentionDecodeOptimized(
        buf.normed,
        block,
        cache,
        pos,
        headDim,
        i
      );

      // Residual
      for (let d = 0; d < dim; d++) {
        buf.hidden[d] += buf.attnOut[d];
      }

      // RMSNorm -> normed buffer
      rmsNormSingleInto(buf.hidden, block.norm2Weight, buf.normed);

      // MLP -> mlpOut buffer
      await this.swigluSingleOptimized(buf.normed, block, i);

      // Residual
      for (let d = 0; d < dim; d++) {
        buf.hidden[d] += buf.mlpOut[d];
      }
    }

    this.cacheSeqLen = pos + 1;

    // Final norm -> normed buffer
    rmsNormSingleInto(buf.hidden, this.normWeight, buf.normed);

    // Output projection -> logits buffer
    matmulFP32Into(buf.normed, this.head, dim, vocabSize, buf.logits);
    return buf.logits;
  }

  private async ternaryMatmulDispatch(
    input: Float32Array,
    layer: TernaryLayer,
    seqLen: number,
    layerName: string
  ): Promise<Float32Array> {
    if (this.gpu) {
      return this.gpu.matmul(input, layerName, seqLen);
    }
    return ternaryMatmul(input, layer, seqLen);
  }

  private async swiglu(
    x: Float32Array,
    block: TransformerBlock,
    seqLen: number,
    blockIdx: number
  ): Promise<Float32Array> {
    const prefix = `block${blockIdx}`;
    const gate = await this.ternaryMatmulDispatch(
      x,
      block.wGate,
      seqLen,
      `${prefix}_wGate`
    );
    const up = await this.ternaryMatmulDispatch(
      x,
      block.wUp,
      seqLen,
      `${prefix}_wUp`
    );

    for (let i = 0; i < gate.length; i++) {
      const g = gate[i];
      const silu = g / (1 + Math.exp(-g));
      gate[i] = silu * up[i];
    }

    return this.ternaryMatmulDispatch(
      gate,
      block.wDown,
      seqLen,
      `${prefix}_wDown`
    );
  }

  /**
   * Optimized SwiGLU MLP using pre-allocated buffers.
   * Writes result to buf.mlpOut.
   */
  private async swigluSingleOptimized(
    x: Float32Array,
    block: TransformerBlock,
    blockIdx: number
  ): Promise<void> {
    const buf = this.decodeBuffers!;
    const prefix = `block${blockIdx}`;

    // Use GPU if available, otherwise CPU with pre-allocated buffers
    if (this.gpu) {
      const gate = await this.gpu.matmul(x, `${prefix}_wGate`, 1);
      const up = await this.gpu.matmul(x, `${prefix}_wUp`, 1);

      for (let i = 0; i < gate.length; i++) {
        const g = gate[i];
        const silu = g / (1 + Math.exp(-g));
        buf.gate[i] = silu * up[i];
      }

      const down = await this.gpu.matmul(buf.gate, `${prefix}_wDown`, 1);
      buf.mlpOut.set(down);
    } else {
      // CPU path with pre-allocated buffers
      ternaryMatmulSingleInto(x, block.wGate, buf.gate);
      ternaryMatmulSingleInto(x, block.wUp, buf.up);

      // SiLU(gate) * up -> gate buffer
      for (let i = 0; i < buf.gate.length; i++) {
        const g = buf.gate[i];
        const silu = g / (1 + Math.exp(-g));
        buf.gate[i] = silu * buf.up[i];
      }

      // Down projection -> mlpOut
      ternaryMatmulSingleInto(buf.gate, block.wDown, buf.mlpOut);
    }
  }

  /**
   * Optimized attention decode using pre-allocated buffers.
   * Writes result to buf.attnOut.
   */
  private async attentionDecodeOptimized(
    x: Float32Array,
    block: TransformerBlock,
    cache: LayerKVCache,
    pos: number,
    headDim: number,
    blockIdx: number
  ): Promise<void> {
    const { nHeads, nKvHeads, maxSeqLen } = this._config;
    const kvDim = nKvHeads * headDim;
    const buf = this.decodeBuffers!;
    const prefix = `block${blockIdx}`;

    // Q and KV projections
    if (this.gpu) {
      const qResult = await this.gpu.matmul(x, `${prefix}_qProj`, 1);
      buf.q.set(qResult);
      const kvResult = await this.gpu.matmul(x, `${prefix}_kvProj`, 1);
      buf.kv.set(kvResult);
    } else {
      ternaryMatmulSingleInto(x, block.qProj, buf.q);
      ternaryMatmulSingleInto(x, block.kvProj, buf.kv);
    }

    // Store K, V at current position
    for (let kh = 0; kh < nKvHeads; kh++) {
      const cacheOffset = kh * maxSeqLen * headDim + pos * headDim;
      const kOffset = kh * headDim;
      const vOffset = kvDim + kh * headDim;
      for (let d = 0; d < headDim; d++) {
        cache.k[cacheOffset + d] = buf.kv[kOffset + d];
        cache.v[cacheOffset + d] = buf.kv[vOffset + d];
      }
    }

    // Apply RoPE
    applyRoPESingle(buf.q, this.ropeCache, pos, nHeads);
    applyRoPEToSinglePos(cache.k, this.ropeCache, pos, nKvHeads, maxSeqLen);

    // Compute attention - reuse scores buffer
    const scale = 1.0 / Math.sqrt(headDim);
    const headsPerKv = nHeads / nKvHeads;
    const seqLen = pos + 1;

    // Zero out attnOut
    buf.attnOut.fill(0);

    for (let h = 0; h < nHeads; h++) {
      const kvHead = Math.floor(h / headsPerKv);
      let maxScore = -Infinity;

      // Compute attention scores - reuse scores buffer (only use first seqLen elements)
      for (let tk = 0; tk < seqLen; tk++) {
        let score = 0;
        const qOffset = h * headDim;
        const kOffset = kvHead * maxSeqLen * headDim + tk * headDim;

        for (let d = 0; d < headDim; d++) {
          score += buf.q[qOffset + d] * cache.k[kOffset + d];
        }
        score *= scale;
        buf.scores[tk] = score;
        if (score > maxScore) maxScore = score;
      }

      // Softmax
      let sumExp = 0;
      for (let tk = 0; tk < seqLen; tk++) {
        buf.scores[tk] = Math.exp(buf.scores[tk] - maxScore);
        sumExp += buf.scores[tk];
      }
      for (let tk = 0; tk < seqLen; tk++) {
        buf.scores[tk] /= sumExp;
      }

      // Weighted sum of values
      const outOffset = h * headDim;
      for (let d = 0; d < headDim; d++) {
        let acc = 0;
        for (let tk = 0; tk < seqLen; tk++) {
          const vOffset = kvHead * maxSeqLen * headDim + tk * headDim;
          acc += buf.scores[tk] * cache.v[vOffset + d];
        }
        buf.attnOut[outOffset + d] = acc;
      }
    }

    // Output projection
    if (this.gpu) {
      const projResult = await this.gpu.matmul(buf.attnOut, `${prefix}_proj`, 1);
      buf.attnOut.set(projResult);
    } else {
      // Use projTemp buffer, then copy back
      ternaryMatmulSingleInto(buf.attnOut, block.proj, buf.projTemp);
      buf.attnOut.set(buf.projTemp);
    }
  }

  private async attentionPrefill(
    x: Float32Array,
    block: TransformerBlock,
    cache: LayerKVCache,
    seqLen: number,
    headDim: number,
    blockIdx: number
  ): Promise<Float32Array> {
    const { dim, nHeads, nKvHeads, maxSeqLen } = this._config;
    const kvDim = nKvHeads * headDim;
    const prefix = `block${blockIdx}`;

    const q = await this.ternaryMatmulDispatch(
      x,
      block.qProj,
      seqLen,
      `${prefix}_qProj`
    );
    const kv = await this.ternaryMatmulDispatch(
      x,
      block.kvProj,
      seqLen,
      `${prefix}_kvProj`
    );

    // Store K, V in cache
    for (let t = 0; t < seqLen; t++) {
      const kvOffset = t * 2 * kvDim;
      for (let kh = 0; kh < nKvHeads; kh++) {
        const cacheOffset = kh * maxSeqLen * headDim + t * headDim;
        const kOffset = kvOffset + kh * headDim;
        const vOffset = kvOffset + kvDim + kh * headDim;
        for (let d = 0; d < headDim; d++) {
          cache.k[cacheOffset + d] = kv[kOffset + d];
          cache.v[cacheOffset + d] = kv[vOffset + d];
        }
      }
    }

    // Apply RoPE
    applyRoPE(q, this.ropeCache, seqLen, nHeads, dim);
    applyRoPEToCache(cache.k, this.ropeCache, seqLen, nKvHeads, maxSeqLen);

    // Compute attention
    const scale = 1.0 / Math.sqrt(headDim);
    const attnOut = new Float32Array(seqLen * dim);
    const headsPerKv = nHeads / nKvHeads;

    for (let h = 0; h < nHeads; h++) {
      const kvHead = Math.floor(h / headsPerKv);

      for (let tq = 0; tq < seqLen; tq++) {
        const scores = new Float32Array(seqLen);
        let maxScore = -Infinity;

        for (let tk = 0; tk <= tq; tk++) {
          let score = 0;
          const qOffset = tq * dim + h * headDim;
          const kOffset = kvHead * maxSeqLen * headDim + tk * headDim;

          for (let d = 0; d < headDim; d++) {
            score += q[qOffset + d] * cache.k[kOffset + d];
          }
          score *= scale;
          scores[tk] = score;
          if (score > maxScore) maxScore = score;
        }

        // Softmax
        let sumExp = 0;
        for (let tk = 0; tk <= tq; tk++) {
          scores[tk] = Math.exp(scores[tk] - maxScore);
          sumExp += scores[tk];
        }
        for (let tk = 0; tk <= tq; tk++) {
          scores[tk] /= sumExp;
        }

        // Weighted sum
        const outOffset = tq * dim + h * headDim;
        for (let d = 0; d < headDim; d++) {
          let acc = 0;
          for (let tk = 0; tk <= tq; tk++) {
            const vOffset = kvHead * maxSeqLen * headDim + tk * headDim;
            acc += scores[tk] * cache.v[vOffset + d];
          }
          attnOut[outOffset + d] = acc;
        }
      }
    }

    return this.ternaryMatmulDispatch(
      attnOut,
      block.proj,
      seqLen,
      `${prefix}_proj`
    );
  }

  private sample(logits: Float32Array, temperature: number): number {
    // Use pre-allocated softmax buffer
    const probs = this.decodeBuffers!.softmax;
    const len = logits.length;
    let maxLogit = -Infinity;

    for (let i = 0; i < len; i++) {
      probs[i] = logits[i] / temperature;
      if (probs[i] > maxLogit) maxLogit = probs[i];
    }

    let sumExp = 0;
    for (let i = 0; i < len; i++) {
      probs[i] = Math.exp(probs[i] - maxLogit);
      sumExp += probs[i];
    }
    for (let i = 0; i < len; i++) {
      probs[i] /= sumExp;
    }

    const r = Math.random();
    let cumsum = 0;
    for (let i = 0; i < len; i++) {
      cumsum += probs[i];
      if (r < cumsum) return i;
    }

    return len - 1;
  }
}
