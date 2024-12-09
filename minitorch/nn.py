from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_h = height // kh
    new_w = width // kw

    tensor = input.contiguous()

    blocked = tensor.view(
        batch,
        channel,
        new_h,
        kh,
        new_w,
        kw,
    )

    blocked = blocked.permute(0, 1, 2, 4, 3, 5)

    blocked = blocked.contiguous()

    pooled = blocked.view(batch, channel, new_h, new_w, kh * kw)

    return pooled, new_h, new_w


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Compute average pooling."""
    pooled_tensor, h_out, w_out = tile(input, kernel)

    return pooled_tensor.mean(dim=4).view(
        pooled_tensor.shape[0], pooled_tensor.shape[1], h_out, w_out
    )


fast_max = FastOps.reduce(operators.max, -float("inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Create a boolean mask showing where maximum values occur along specified dimension.

    Args:
        input: Input tensor
        dim: Dimension to find max values along

    Returns:
        Boolean tensor with True at max value positions

    """
    # Get max values along dimension
    max_vals = fast_max(input, dim)

    # Create mask by comparing input to max values
    return input == max_vals


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, dimension: Tensor) -> Tensor:
        """Forward pass computes max values along given dimension.

        Args:
            ctx: Context for backprop
            input_tensor: Tensor to reduce
            dimension: Which dimension to reduce along

        """
        dim = int(dimension.item())
        ctx.save_for_backward(input_tensor, dim)
        return fast_max(input_tensor, dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass routes gradients to max value positions.

        Args:
            ctx: Context with saved values
            grad_output: Incoming gradient

        Returns:
            Tuple of (input gradient, dimension gradient)

        """
        input_tensor, dim = ctx.saved_values
        max_positions = argmax(input_tensor, dim)
        return max_positions * grad_output, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Take maximum values along specified dimension.

    Args:
        input: Input tensor
        dim: Dimension to reduce

    Returns:
        Tensor of max values

    """
    return Max.apply(input, tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax probabilities along specified dimension.

    Args:
        input: Input tensor
        dim: Dimension for softmax

    Returns:
        Tensor of softmax probabilities

    """
    # Subtract max for numerical stability
    max_vals = max(input, dim)
    shifted = input - max_vals

    # Compute normalized exponentials
    exp_vals = shifted.exp()
    sum_exp = exp_vals.sum(dim)

    return exp_vals / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log softmax values along specified dimension.

    Args:
        input: Input tensor
        dim: Dimension for logsoftmax

    Returns:
        Tensor of log softmax values

    """
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D max pooling with specified kernel size.

    Args:
        input: Input tensor (batch x channels x height x width)
        kernel: (kernel_height, kernel_width) tuple

    Returns:
        Pooled output tensor

    """
    batch, channels, _, _ = input.shape
    kh, kw = kernel

    # Reshape input into patches
    tiled, new_height, new_width = tile(input, (kh, kw))

    # Take max over patch dimension
    pooled = max(tiled, dim=4).contiguous()

    # Reshape to final output size
    return pooled.view(batch, channels, new_height, new_width)


def dropout(input: Tensor, p: float = 0.5, ignore: bool = False) -> Tensor:
    """Apply dropout with specified probability.

    Args:
        input: Input tensor
        p: Dropout probability
        ignore: If True, return input unchanged

    Returns:
        Output with dropout applied

    """
    if ignore or p <= 0.0:
        return input
    if p >= 1.0:
        return input.zeros(input.shape)

    # Generate and apply dropout mask
    keep_mask = rand(input.shape) > p
    return input * keep_mask
