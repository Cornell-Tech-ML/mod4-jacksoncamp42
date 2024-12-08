from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the sum of two floats."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the gradient of the addition function."""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the natural logarithm of a float."""
        ctx.save_for_backward(a)
        return float(operators.log(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the gradient of the log function."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for multiplication.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: The product of a and b.

        """
        ctx.save_for_backward(a, b)
        return float(operators.mul(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for multiplication.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to inputs a and b.

        """
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for inverse (1/x).

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): Input value.

        Returns:
        -------
            float: The inverse of a (1/a).

        """
        ctx.save_for_backward(a)
        return float(operators.inv(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for inverse.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: Gradient with respect to input a.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for negation.

        Args:
        ----
            ctx (Context): The context (unused in this case).
            a (float): Input value.

        Returns:
        -------
            float: The negation of a (-a).

        """
        return float(operators.neg(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for negation.

        Args:
        ----
            ctx (Context): The context (unused in this case).
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: Gradient with respect to input a.

        """
        return operators.neg(d_output)


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for sigmoid function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): Input value.

        Returns:
        -------
            float: The sigmoid of a.

        """
        sigmoid_val = float(operators.sigmoid(a))
        ctx.save_for_backward(sigmoid_val)
        return sigmoid_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for sigmoid function.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: Gradient with respect to input a.

        """
        (sigmoid_val,) = ctx.saved_values
        return operators.mul(
            d_output,
            operators.mul(sigmoid_val, operators.add(1.0, operators.neg(sigmoid_val))),
        )


class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for ReLU function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): Input value.

        Returns:
        -------
            float: The ReLU of a (max(0, a)).

        """
        ctx.save_for_backward(a)
        return float(operators.relu(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for ReLU function.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: Gradient with respect to input a.

        """
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Forward pass for exponential function.

        Args:
        ----
            ctx (Context): The context to save values for backward pass.
            a (float): Input value.

        Returns:
        -------
            float: The exponential of a (e^a).

        """
        exp_val = float(operators.exp(a))
        ctx.save_for_backward(exp_val)
        return exp_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Backward pass for exponential function.

        Args:
        ----
            ctx (Context): The context with saved values from forward pass.
            d_output (float): The gradient of the output.

        Returns:
        -------
            float: Gradient with respect to input a.

        """
        (exp_val,) = ctx.saved_values
        return d_output * exp_val


class LT(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for less than comparison.

        Args:
        ----
            ctx (Context): The context (unused in this case).
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: 1.0 if a < b, else 0.0.

        """
        return float(operators.lt(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for less than comparison.

        Args:
        ----
            ctx (Context): The context (unused in this case).
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to inputs a and b (always 0).

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Forward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context (unused in this case).
            a (float): First input.
            b (float): Second input.

        Returns:
        -------
            float: 1.0 if a == b, else 0.0.

        """
        return float(operators.eq(a, b))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Backward pass for equality comparison.

        Args:
        ----
            ctx (Context): The context (unused in this case).
            d_output (float): The gradient of the output.

        Returns:
        -------
            Tuple[float, float]: Gradients with respect to inputs a and b (always 0).

        """
        return 0.0, 0.0
