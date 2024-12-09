from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Protocol, Tuple

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    vals_up = [v for v in vals]
    vals_down = [v for v in vals]
    vals_up[arg] = vals_up[arg] + epsilon
    vals_down[arg] = vals_down[arg] - epsilon
    return (f(*vals_up) - f(*vals_down)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """Checks if this variable is a leaf in the computation graph."""
        ...

    def is_constant(self) -> bool:
        """Checks if this variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute gradients for this variable's parents."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    visited = set()
    topo_order: List[Variable] = []

    def dfs(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        if not v.is_leaf():
            for m in v.parents:
                if not m.is_constant():
                    dfs(m)
        visited.add(v.unique_id)
        topo_order.insert(0, v)

    dfs(variable)
    return topo_order


def backpropagate(variable: Variable, deriv: Any) -> None:  # noqa: D417
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv: The derivative of the final output with respect to the `variable`

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    queue = topological_sort(variable)
    derivs = {}
    derivs[variable.unique_id] = deriv

    for v in queue:
        deriv = derivs[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(deriv)
        else:
            for v, d in v.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivs.setdefault(v.unique_id, 0.0)
                derivs[v.unique_id] = derivs[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the values saved for backward computation."""
        return self.saved_values
