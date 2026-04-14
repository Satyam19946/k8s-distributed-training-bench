# Phase 2: Baseline Single-Node PyTorchJob with MLflow Tracking

## Goal
Validate the full pipeline end-to-end on a single node before introducing
multi-node complexity in Phase 3. Establish baseline throughput numbers with
world_size=1 so that AllReduce overhead is measurable by diff in Phase 3.

## Components

### Registry (host-side)
A Docker registry runs on the host at 192.168.56.1:5000, managed via
docker-compose. All cluster nodes are configured to pull from it as an
insecure registry via /etc/containerd/config.toml.

Start: `docker compose -f ../registry/docker-compose.yml up -d`

### MLflow
Deployed in the `mlflow` namespace, pinned to worker1 via nodeSelector.
Persistent storage at /data/mlflow on worker1 via hostPath volume.
Backend store: SQLite. Artifact root: local filesystem.

UI: http://192.168.56.11:30500
In-cluster URI: http://mlflow.mlflow.svc.cluster.local:5000

### Training Script (trainer/train.py)
- PyTorch DDP, Gloo backend, synthetic data (no dataset download)
- Logs per-step: loss, step_time_ms, samples_per_sec
- Logs summary: avg_step_time_ms, avg_samples_per_sec
- Tags each run with: world_size, backend, batch_size, cni_plugin
- Fails fast if MLflow is unreachable (abort before wasting compute)
- Only rank 0 logs to MLflow

### Container Image
Base: python:3.11-slim
Packages: torch==2.3.1+cpu, mlflow==2.13.0
Registry: 192.168.56.1:5000/train:v1

Rebuild and push:
```bash
cd trainer
docker build -t 192.168.56.1:5000/train:v1 .
docker push 192.168.56.1:5000/train:v1
```

### PyTorchJob
Single master, no workers (world_size=1).
Manifest: pytorchjob-baseline.yaml

Run: `kubectl apply -f pytorchjob-baseline.yaml`
Watch: `kubectl get pytorchjob baseline-single-node -w`
Logs: `kubectl logs -f baseline-single-node-master-0`
Clean up: `kubectl delete pytorchjob baseline-single-node`

## Baseline Results
| Metric | Value |
|---|---|
| world_size | 1 |
| avg_step_time_ms | 1.9 |
| avg_samples_per_sec | 33724 |
| AllReduce cost | 0ms (no-op at world_size=1) |

These numbers are the Phase 3 comparison baseline. The break-even point for
distributed training to be worthwhile is step_time < 2 × 1.9ms = 3.8ms.
If Phase 3 step_time exceeds 3.8ms, AllReduce overhead is eating the
parallelism gains.

## Key Learnings
- OCI image spec is what makes images portable across runtimes (Docker build,
  containerd pull — same image format)
- Registry port binding is fixed at `docker run` time — use docker-compose to
  avoid losing config across Docker daemon restarts
- kubeconfig current-context must be explicitly set after pulling admin.conf
- Only rank 0 touches MLflow — other ranks are unaware of it entirely
- Fail fast on MLflow unavailability — an untracked experiment run is useless
  for a benchmarking project