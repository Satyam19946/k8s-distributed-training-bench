import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import mlflow
import requests
import sys

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

def assert_mlflow_reachable(tracking_uri):
    """
    Probe the MLflow health endpoint before starting any work.
    Fails fast with a clear error if the server is unreachable.
    """
    try:
        health_url = tracking_uri.rstrip("/") + "/health"
        resp = requests.get(health_url, timeout=5)
        resp.raise_for_status()
        print(f"MLflow reachable at {tracking_uri}")
    except Exception as e:
        print(f"FATAL: MLflow server unreachable at {tracking_uri}: {e}")
        print("Aborting — no point running an untracked experiment.")
        sys.exit(1)

def main():
    rank, world_size = init_distributed()

    # Rank 0 validates MLflow before any training work starts.
    # If it exits here, the Training Operator marks the job as failed —
    # which is the correct observable outcome.
    if rank == 0:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        assert_mlflow_reachable(tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("distributed-training-bench")
        mlflow.start_run()
        mlflow.log_params({
            "world_size": world_size,
            "backend":    "gloo",
            "batch_size": BATCH_SIZE,
            "num_steps":  NUM_STEPS,
            "input_dim":  INPUT_DIM,
            "cni_plugin": os.environ.get("CNI_PLUGIN", "flannel"),
        })

    model = SmallNet()
    model = nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    step_times = []

    for step in range(NUM_STEPS):
        inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
        labels = torch.randint(0, 10, (BATCH_SIZE,))

        t0 = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        step_time_ms = (time.perf_counter() - t0) * 1000
        step_times.append(step_time_ms)
        samples_per_sec = (BATCH_SIZE * world_size) / (step_time_ms / 1000)

        if rank == 0:
            mlflow.log_metrics({
                "loss":            loss.item(),
                "step_time_ms":    step_time_ms,
                "samples_per_sec": samples_per_sec,
            }, step=step)

            if step % 10 == 0:
                print(f"step={step:3d}  loss={loss.item():.4f}  "
                      f"step_time={step_time_ms:.1f}ms  "
                      f"samples/sec={samples_per_sec:.1f}")

    if rank == 0:
        avg_step_ms = sum(step_times) / len(step_times)
        avg_samples = (BATCH_SIZE * world_size) / (avg_step_ms / 1000)
        mlflow.log_metrics({
            "avg_step_time_ms":    avg_step_ms,
            "avg_samples_per_sec": avg_samples,
        }, step=NUM_STEPS)
        mlflow.end_run()
        print(f"\nDone. avg_step_time={avg_step_ms:.1f}ms  avg_samples/sec={avg_samples:.1f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
