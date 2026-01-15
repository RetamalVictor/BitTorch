/**
 * Embedding - Token embedding lookup
 *
 * @module core/Embedding
 */

/**
 * Look up embeddings for a sequence of tokens.
 *
 * @param tokenIds - Token IDs [seqLen]
 * @param embedding - Embedding matrix [vocabSize, dim]
 * @param dim - Embedding dimension
 * @returns Embedded vectors [seqLen, dim]
 */
export function embeddingLookup(
  tokenIds: number[],
  embedding: Float32Array,
  dim: number
): Float32Array {
  const seqLen = tokenIds.length;
  const result = new Float32Array(seqLen * dim);

  for (let t = 0; t < seqLen; t++) {
    const tokenId = tokenIds[t];
    const srcOffset = tokenId * dim;
    const dstOffset = t * dim;
    for (let d = 0; d < dim; d++) {
      result[dstOffset + d] = embedding[srcOffset + d];
    }
  }

  return result;
}

/**
 * Look up embedding for a single token.
 *
 * @param tokenId - Token ID
 * @param embedding - Embedding matrix [vocabSize, dim]
 * @param dim - Embedding dimension
 * @returns Embedded vector [dim]
 */
export function embeddingLookupSingle(
  tokenId: number,
  embedding: Float32Array,
  dim: number
): Float32Array {
  const result = new Float32Array(dim);
  const srcOffset = tokenId * dim;
  for (let d = 0; d < dim; d++) {
    result[d] = embedding[srcOffset + d];
  }
  return result;
}
