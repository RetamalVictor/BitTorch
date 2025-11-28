#include <torch/extension.h>

namespace bittorch {

// Dummy function to verify the extension is working
torch::Tensor dummy(torch::Tensor input) {
  // Simply return a clone of the input
  return input.clone();
}

} // namespace bittorch
