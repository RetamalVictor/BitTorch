"""Test that the C++ extension loads and works correctly."""

import pytest
import torch


def test_import_bittorch():
    """Test that bittorch can be imported."""
    import bittorch
    assert bittorch.__version__ == "0.0.1"


def test_has_cuda_ext():
    """Test that we can check for CUDA extension availability."""
    import bittorch
    # This should return True if extension is compiled, False otherwise
    result = bittorch.has_cuda_ext()
    assert isinstance(result, bool)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)
def test_dummy_function():
    """Test the dummy function from the C++ extension."""
    import bittorch

    if not bittorch.has_cuda_ext():
        pytest.skip("CUDA extension not compiled")

    # Create a test tensor
    x = torch.randn(4, 4, device="cuda")

    # Call the dummy function
    y = bittorch._C.dummy(x)

    # Should return a clone of the input
    assert torch.allclose(x, y)
    assert x.data_ptr() != y.data_ptr()  # Should be a different tensor


def test_dummy_function_cpu():
    """Test the dummy function works on CPU tensors too."""
    import bittorch

    if not bittorch.has_cuda_ext():
        pytest.skip("CUDA extension not compiled")

    # Create a test tensor on CPU
    x = torch.randn(4, 4)

    # Call the dummy function
    y = bittorch._C.dummy(x)

    # Should return a clone of the input
    assert torch.allclose(x, y)
    assert x.data_ptr() != y.data_ptr()
