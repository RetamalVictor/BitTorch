#pragma once

#include <cuda_runtime.h>

namespace bittorch {

// Check CUDA errors
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Ceiling division
template <typename T>
__host__ __device__ inline T ceildiv(T a, T b) {
  return (a + b - 1) / b;
}

} // namespace bittorch
