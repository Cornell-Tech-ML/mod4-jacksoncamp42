from typing import Any, Tuple, TypeVar

import numpy as np
from numba import njit as _njit
from numba import prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Storage,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Just-in-time compile a function using Numba.

    Args:
        fn (Fn): The function to be compiled.
        **kwargs: Additional keyword arguments for Numba's njit.

    Returns:
        Fn: The compiled function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right

    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    # s1 = input_strides
    # s2 = weight_strides

    # TODO: Implement for Task 4.1.
    # Parallelize over independent batch and output channel computations
    for b in prange(batch):
        for oc in prange(out_channels):
            # These loops can be parallelized as they compute different output elements
            for w in prange(out_width):
                out_index = np.zeros(3, np.int32)
                out_index[0] = b
                out_index[1] = oc
                out_index[2] = w
                out_pos = index_to_position(out_index, out_strides)

                acc = 0.0
                # These loops must be sequential as they contribute to same accumulator
                for ic in range(in_channels):
                    for k in range(kw):
                        if reverse:
                            w_pos = w - (kw - 1) + k
                        else:
                            w_pos = w + k

                        if 0 <= w_pos < width:
                            in_index = np.zeros(3, np.int32)
                            in_index[0] = b
                            in_index[1] = ic
                            in_index[2] = w_pos
                            in_pos = index_to_position(in_index, input_strides)

                            weight_index = np.zeros(3, np.int32)
                            weight_index[0] = oc
                            weight_index[1] = ic
                            weight_index[2] = k if not reverse else (kw - 1 - k)
                            weight_pos = index_to_position(weight_index, weight_strides)

                            acc += input[in_pos] * weight[weight_pos]

                out[out_pos] = acc


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the convolution operation.

        Args:
        ----
            ctx : Context
                The context object containing saved values.
            grad_output : Tensor
                The gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]
                Gradients with respect to the input and weight tensors.

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
    ----
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right

    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]

    # TODO: Implement for Task 4.2.
    batch_, out_channels, height_, width_ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape
    # Parallelize over independent batch and output channel computations
    for b in prange(batch):
        for oc in prange(out_channels):
            # These loops can be parallelized as they compute different output elements
            for h in prange(height_):
                for w in prange(width_):
                    out_index = np.zeros(4, np.int32)
                    out_index[0] = b
                    out_index[1] = oc
                    out_index[2] = h
                    out_index[3] = w
                    out_pos = index_to_position(out_index, out_strides)

                    acc = 0.0
                    # These loops must be sequential as they contribute to same accumulator
                    for ic in range(in_channels):
                        for kh_pos in range(kh):
                            for kw_pos in range(kw):
                                if reverse:
                                    h_pos = h - (kh - 1) + kh_pos
                                    w_pos = w - (kw - 1) + kw_pos
                                else:
                                    h_pos = h + kh_pos
                                    w_pos = w + kw_pos

                                if 0 <= h_pos < height and 0 <= w_pos < width:
                                    # Use cached strides for faster access
                                    in_pos = (
                                        b * s10 + ic * s11 + h_pos * s12 + w_pos * s13
                                    )

                                    # Use cached strides for faster access
                                    weight_pos = (
                                        oc * s20
                                        + ic * s21
                                        + (kh_pos if not reverse else (kh - 1 - kh_pos))
                                        * s22
                                        + (kw_pos if not reverse else (kw - 1 - kw_pos))
                                        * s23
                                    )

                                    acc += input[in_pos] * weight[weight_pos]

                    out[out_pos] = acc


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the convolution operation.

        Args:
        ----
            ctx : Context
                The context object containing saved values.
            grad_output : Tensor
                The gradient of the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]
                Gradients with respect to the input and weight tensors.

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
