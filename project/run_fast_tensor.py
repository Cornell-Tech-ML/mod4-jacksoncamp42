import random
import time

import numba
import numpy as np

import minitorch

datasets = minitorch.datasets
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)
# Global dictionary to store timing information
timing_stats = {"start_time": None, "epoch_times": [], "last_epoch_time": None}


def timing_log_fn(epoch, total_loss, correct, losses):
    current_time = time.time()

    # Initialize start time if first epoch
    if timing_stats["start_time"] is None:
        timing_stats["start_time"] = current_time
        timing_stats["last_epoch_time"] = current_time
        time_per_epoch = 0
    else:
        # Calculate time for this epoch and divide by BATCH size (10)
        time_per_epoch = (current_time - timing_stats["last_epoch_time"]) / 10
        timing_stats["epoch_times"].append(time_per_epoch)
        timing_stats["last_epoch_time"] = current_time

    if epoch % 10 == 0 or epoch == max_epochs:
        avg_time = (
            np.mean(timing_stats["epoch_times"]) if timing_stats["epoch_times"] else 0
        )
        print(
            f"Epoch {epoch:3d} | Loss {total_loss:10.2f} | Correct {correct:4d} | Time/epoch {avg_time:5.3f}s"
        )


def run_training_benchmark():
    datasets = ["simple", "xor", "split"]
    backends = {
        "cpu": FastTensorBackend,
        "gpu": GPUBackend if numba.cuda.is_available() else None,
    }

    results = {}

    for dataset_name in datasets:
        results[dataset_name] = {}
        print(f"\nTraining on {dataset_name} dataset")
        print("=" * 80)

        for backend_name, backend in backends.items():
            if backend is None:
                continue

            print(f"\nUsing {backend_name.upper()} backend")
            print("-" * 40)

            # Reset timing stats
            timing_stats.clear()
            timing_stats.update(
                {"start_time": None, "epoch_times": [], "last_epoch_time": None}
            )

            # Create dataset
            if dataset_name == "xor":
                data = minitorch.datasets["Xor"](PTS)
            elif dataset_name == "simple":
                data = minitorch.datasets["Simple"].simple(PTS)
            else:  # split
                data = minitorch.datasets["Split"](PTS)

            # Train
            FastTrain(HIDDEN, backend=backend).train(data, RATE, log_fn=timing_log_fn)

            # Store results
            results[dataset_name][backend_name] = {
                "avg_time_per_epoch": np.mean(timing_stats["epoch_times"]),
                "total_time": time.time() - timing_stats["start_time"],
            }

    # Print summary
    print("\nFinal Summary:")
    print("=" * 80)
    print(f"{'Dataset':<10} | {'Backend':<6} | {'Time/Epoch':<12}")
    print("-" * 80)

    for dataset in results:
        for backend in results[dataset]:
            avg_time = results[dataset][backend]["avg_time_per_epoch"]
            print(f"{dataset:<10} | {backend:<6} | {avg_time:>8.3f}s")
        print("-" * 40)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


def RParam(*shape, backend):
    r = minitorch.rand(shape, backend=backend) - 0.5
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden, backend):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden, backend)
        self.layer2 = Linear(hidden, hidden, backend)
        self.layer3 = Linear(hidden, 1, backend)

    def forward(self, x):
        # TODO: Implement for Task 3.5.
        # Similar to previous implementation but using fast operations
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size, backend):
        super().__init__()
        self.weights = RParam(in_size, out_size, backend=backend)
        s = minitorch.zeros((out_size,), backend=backend)
        s = s + 0.1
        self.bias = minitorch.Parameter(s)
        self.out_size = out_size

    def forward(self, x):
        # TODO: Implement for Task 3.5.
        # Use efficient matrix multiplication
        batch, in_size = x.shape
        w = self.weights.value.view(in_size, self.out_size)

        # Efficient matmul implementation
        out = x @ w

        # Add bias - broadcasting will handle batch dimension
        return out + self.bias.value


class FastTrain:
    def __init__(self, hidden_layers, backend=FastTensorBackend):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers, backend)
        self.backend = backend

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.model = Network(self.hidden_layers, self.backend)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        BATCH = 10
        losses = []

        for epoch in range(max_epochs):
            total_loss = 0.0
            c = list(zip(data.X, data.y))
            random.shuffle(c)
            X_shuf, y_shuf = zip(*c)

            for i in range(0, len(X_shuf), BATCH):
                optim.zero_grad()
                X = minitorch.tensor(X_shuf[i : i + BATCH], backend=self.backend)
                y = minitorch.tensor(y_shuf[i : i + BATCH], backend=self.backend)
                # Forward

                out = self.model.forward(X).view(y.shape[0])
                prob = (out * y) + (out - 1.0) * (y - 1.0)
                loss = -prob.log()
                (loss / y.shape[0]).sum().view(1).backward()

                total_loss = loss.sum().view(1)[0]

                # Update
                optim.step()

            losses.append(total_loss)
            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                X = minitorch.tensor(data.X, backend=self.backend)
                y = minitorch.tensor(data.y, backend=self.backend)
                out = self.model.forward(X).view(y.shape[0])
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points")
    parser.add_argument("--HIDDEN", type=int, default=100, help="number of hiddens")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate")
    parser.add_argument("--BACKEND", default="cpu", help="backend mode")
    parser.add_argument("--DATASET", default="simple", help="dataset")
    parser.add_argument("--BENCHMARK", action="store_true", help="run full benchmark")

    args = parser.parse_args()

    PTS = args.PTS
    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    if args.BENCHMARK:
        run_training_benchmark()
    else:
        # Original training code
        if args.DATASET == "xor":
            data = minitorch.datasets["Xor"](PTS)
        elif args.DATASET == "simple":
            data = minitorch.datasets["Simple"](PTS)
        elif args.DATASET == "split":
            data = minitorch.datasets["Split"](PTS)

        backend = FastTensorBackend if args.BACKEND != "gpu" else GPUBackend
        FastTrain(HIDDEN, backend=backend).train(data, RATE, log_fn=timing_log_fn)
