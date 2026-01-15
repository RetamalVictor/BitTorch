/**
 * BPE Tokenizer for Browser
 *
 * Implements Byte-Pair Encoding compatible with HuggingFace tokenizers.
 *
 * @module tokenizers/BPETokenizer
 */

/** Configuration for BPE tokenizer */
export interface TokenizerConfig {
  vocab: Map<string, number>;
  merges: Array<[string, string]>;
  decoder: Map<number, string>;
  unkToken: string;
  unkId: number;
}

/**
 * Byte-Pair Encoding tokenizer.
 *
 * Compatible with HuggingFace tokenizer.json format.
 *
 * @example
 * ```typescript
 * const tokenizer = await BPETokenizer.fromUrl('/model/tokenizer.json');
 *
 * // Encode text to token IDs
 * const ids = tokenizer.encode('Hello, world!');
 *
 * // Decode token IDs back to text
 * const text = tokenizer.decode(ids);
 * ```
 */
export class BPETokenizer {
  private vocab: Map<string, number>;
  private merges: Array<[string, string]>;
  private mergeRanks: Map<string, number>;
  private decoder: Map<number, string>;
  private unkId: number;

  private constructor(config: TokenizerConfig) {
    this.vocab = config.vocab;
    this.merges = config.merges;
    this.decoder = config.decoder;
    this.unkId = config.unkId;

    // Build merge ranks for efficient lookup
    this.mergeRanks = new Map();
    for (let i = 0; i < this.merges.length; i++) {
      const [a, b] = this.merges[i];
      this.mergeRanks.set(`${a} ${b}`, i);
    }
  }

  /**
   * Load tokenizer from HuggingFace JSON format.
   *
   * @param url - URL to tokenizer.json file
   * @returns Promise resolving to BPETokenizer instance
   */
  static async fromUrl(url: string): Promise<BPETokenizer> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch tokenizer: ${response.statusText}`);
    }
    const data = await response.json();
    return BPETokenizer.fromJSON(data);
  }

  /**
   * Create tokenizer from parsed JSON.
   *
   * @param data - Parsed tokenizer.json contents
   * @returns BPETokenizer instance
   */
  static fromJSON(data: Record<string, unknown>): BPETokenizer {
    const model = data.model as Record<string, unknown>;
    if (!model || model.type !== "BPE") {
      throw new Error("Expected BPE tokenizer");
    }

    // Build vocab map
    const vocab = new Map<string, number>();
    const vocabObj = model.vocab as Record<string, number>;
    for (const [token, id] of Object.entries(vocabObj)) {
      vocab.set(token, id);
    }

    // Build decoder (id -> token)
    const decoder = new Map<number, string>();
    for (const [token, id] of vocab) {
      decoder.set(id, token);
    }

    // Parse merges - can be either ["a", "b"] arrays or "a b" strings
    const merges: Array<[string, string]> = [];
    const rawMerges = (model.merges as Array<[string, string] | string>) || [];
    for (const merge of rawMerges) {
      if (Array.isArray(merge) && merge.length === 2) {
        // Already in [a, b] format
        merges.push([merge[0], merge[1]]);
      } else if (typeof merge === "string") {
        // "a b" format - split on space
        const parts = merge.split(" ");
        if (parts.length === 2) {
          merges.push([parts[0], parts[1]]);
        }
      }
    }

    // Get unknown token
    const unkToken = (model.unk_token as string) || "<unk>";
    const unkId = vocab.get(unkToken) ?? 0;

    return new BPETokenizer({
      vocab,
      merges,
      decoder,
      unkToken,
      unkId,
    });
  }

  /**
   * Get vocabulary size.
   */
  get vocabSize(): number {
    return this.vocab.size;
  }

  /**
   * Encode text to token IDs.
   *
   * @param text - Input text to tokenize
   * @returns Array of token IDs
   */
  encode(text: string): number[] {
    if (text.length === 0) return [];

    // Split into words (simple whitespace-based for now)
    // HuggingFace uses Ġ prefix for tokens that follow whitespace
    const words = this.preTokenize(text);

    const tokens: number[] = [];
    for (const word of words) {
      const wordTokens = this.encodeWord(word);
      tokens.push(...wordTokens);
    }

    return tokens;
  }

  /**
   * Pre-tokenize: split on whitespace while preserving it as prefix.
   */
  private preTokenize(text: string): string[] {
    const words: string[] = [];
    let current = "";
    let atWordStart = true;

    for (let i = 0; i < text.length; i++) {
      const char = text[i];

      if (char === " " || char === "\n" || char === "\t") {
        if (current.length > 0) {
          words.push(current);
          current = "";
        }
        // Next non-space char will have Ġ prefix
        atWordStart = true;
      } else {
        if (atWordStart && words.length > 0) {
          // Add Ġ prefix for words after whitespace
          current = "Ġ" + char;
        } else {
          current += char;
        }
        atWordStart = false;
      }
    }

    if (current.length > 0) {
      words.push(current);
    }

    return words;
  }

  /**
   * Encode a single word using BPE.
   */
  private encodeWord(word: string): number[] {
    // Start with individual characters
    let tokens: string[] = Array.from(word);

    // Apply BPE merges
    while (tokens.length > 1) {
      // Find the merge with lowest rank
      let bestMergeIdx = -1;
      let bestRank = Infinity;

      for (let i = 0; i < tokens.length - 1; i++) {
        const pair = `${tokens[i]} ${tokens[i + 1]}`;
        const rank = this.mergeRanks.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestMergeIdx = i;
        }
      }

      if (bestMergeIdx === -1) {
        // No more merges possible
        break;
      }

      // Apply the merge
      const merged = tokens[bestMergeIdx] + tokens[bestMergeIdx + 1];
      tokens = [
        ...tokens.slice(0, bestMergeIdx),
        merged,
        ...tokens.slice(bestMergeIdx + 2),
      ];
    }

    // Convert tokens to IDs
    return tokens.map((t) => this.vocab.get(t) ?? this.unkId);
  }

  /**
   * Decode token IDs to text.
   *
   * @param ids - Array of token IDs
   * @returns Decoded text string
   */
  decode(ids: number[]): string {
    const tokens = ids.map((id) => this.decoder.get(id) ?? "");
    let text = tokens.join("");

    // Replace special tokens with their actual characters
    text = this.cleanText(text);

    return text;
  }

  /**
   * Decode a single token ID to string.
   *
   * @param id - Token ID
   * @returns Token string
   */
  decodeToken(id: number): string {
    const token = this.decoder.get(id) ?? "";
    return this.cleanText(token);
  }

  /**
   * Clean up special token encodings in text.
   */
  private cleanText(text: string): string {
    return text
      .replace(/Ġ/g, " ")      // Space prefix
      .replace(/Ċ/g, "\n")     // Newline
      .replace(/ĉ/g, "\t")     // Tab
      .replace(/@-@/g, "-")    // Hyphen encoding
      .replace(/Ã/g, "")       // Remove stray encoding artifacts
      .replace(/â/g, "'");     // Smart quote encoding
  }

  /**
   * Get token ID for a single token string.
   *
   * @param token - Token string
   * @returns Token ID or unknown ID if not found
   */
  tokenToId(token: string): number {
    return this.vocab.get(token) ?? this.unkId;
  }

  /**
   * Get token string for an ID.
   *
   * @param id - Token ID
   * @returns Token string or empty string if not found
   */
  idToToken(id: number): string {
    return this.decoder.get(id) ?? "";
  }
}
