import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def init_distributed():
    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "23456")

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.fc(x)


BATCH_SIZE = 64
NUM_STEPS  = 100
INPUT_DIM  = 512


def init_mlflow(rank, world_size):
    if not MLFLOW_AVAILABLE or rank != 0:
        return False
    try:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("distributed-training-bench")
        mlflow.start_run()
        mlflow.log_params({
            "world_size": world_size,
            "backend":    "gloo",
            "batch_size": BATCH_SIZE,
            "num_steps":  NUM_STEPS,
            "input_dim":  INPUT_DIM,
            "phase":      "03-multi-node",
            "cni_plugin": os.environ.get("CNI_PLUGIN", "flannel"),
        })
        return True
    except Exception as e:
        print(f"[rank {rank}] MLflow init failed: {e} — continuing without tracking")
        return False


def main():
    rank, world_size = init_distributed()
    print(f"[rank {rank}] init_process_group complete — world_size={world_size}")

    model = SmallNet()
    model = nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    mlflow_active = init_mlflow(rank, world_size)

    step_times = []

    for step in range(NUM_STEPS):
        inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
        labels = torch.randint(0, 10, (BATCH_SIZE,))

        t0 = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()      # DDP fires AllReduce here — blocks until all ranks sync
        optimizer.step()

        t1 = time.perf_counter()
        step_time_ms = (t1 - t0) * 1000
        step_times.append(step_time_ms)

        if rank == 0 and step % 10 == 0:
            samples_per_sec = (BATCH_SIZE / ((t1 - t0))) * world_size
            print(f"[rank {rank}] step={step:3d}  loss={loss.item():.4f}  "
                  f"step_time={step_time_ms:.2f}ms  samples/sec={samples_per_sec:.0f}")

    # --- Summary (rank 0 only) ---
    if rank == 0:
        avg_step_ms     = sum(step_times) / len(step_times)
        avg_samples_sec = (BATCH_SIZE / (avg_step_ms / 1000)) * world_size

        print(f"\n[rank {rank}] === Phase 3 Results ===")
        print(f"[rank {rank}] avg_step_time     = {avg_step_ms:.2f}ms")
        print(f"[rank {rank}] avg_samples_per_sec = {avg_samples_sec:.0f}")
        print(f"[rank {rank}] break-even threshold = 3.80ms")
        print(f"[rank {rank}] verdict: {'BELOW break-even (communication OK)' if avg_step_ms < 3.8 else 'ABOVE break-even (communication dominant)'}")

        if mlflow_active:
            mlflow.log_metrics({
                "avg_step_time_ms":    avg_step_ms,
                "avg_samples_per_sec": avg_samples_sec,
            })
            mlflow.end_run()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
