import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor

from .strategies import assert_close
from .tensor_strategies import tensors


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    # Test max reduction along the last dimension (dim=2)
    out = minitorch.max(t, dim=2)

    # Verify the shape of the output
    assert out.shape == (2, 3, 1), "Shape mismatch for max reduction along dim=2."

    # Verify the correctness of the max values
    for b in range(2):
        for c in range(3):
            expected = max([t[b, c, k] for k in range(4)])
            assert_close(out[b, c, 0], expected)

    # Test max reduction along the middle dimension (dim=1)
    out = minitorch.max(t, dim=1)

    # Verify the shape of the output
    assert out.shape == (2, 1, 4), "Shape mismatch for max reduction along dim=1."

    # Verify the correctness of the max values
    for b in range(2):
        for w in range(4):
            expected = max([t[b, c, w] for c in range(3)])
            assert_close(out[b, 0, w], expected)

    # Test max reduction along the first dimension (dim=0)
    out = minitorch.max(t, dim=0)

    # Verify the shape of the output
    assert out.shape == (1, 3, 4), "Shape mismatch for max reduction along dim=0."

    # Verify the correctness of the max values
    for c in range(3):
        for w in range(4):
            expected = max([t[b, c, w] for b in range(2)])
            assert_close(out[0, c, w], expected)

    # Gradient checks for max reduction
    # Add small random noise to avoid issues with duplicate max values
    perturbed_t0 = t + minitorch.rand(t.shape) * 1e-5  # For dim=0
    perturbed_t1 = t + minitorch.rand(t.shape) * 1e-5  # For dim=1
    perturbed_t2 = t + minitorch.rand(t.shape) * 1e-5  # For dim=2

    # Perform gradient checks
    minitorch.grad_check(lambda t: minitorch.max(t, dim=0), perturbed_t0)
    minitorch.grad_check(lambda t: minitorch.max(t, dim=1), perturbed_t1)
    minitorch.grad_check(lambda t: minitorch.max(t, dim=2), perturbed_t2)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)
