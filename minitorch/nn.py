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
