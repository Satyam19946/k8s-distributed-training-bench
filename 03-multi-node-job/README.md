# Phase 3: Multi-Node Distributed Training

## What is Distributed Training

When a model is too large to fit in one GPU's memory, or training on a single
machine is too slow, you distribute the work across multiple processes — each
running on a separate CPU or GPU, potentially on separate physical machines.

The dominant strategy for this is **Data Parallelism**: every worker holds a
full copy of the model, but each sees a different slice of the batch. After each
worker computes gradients on its slice, those gradients are averaged across all
workers so every model copy receives the same update. The models stay identical
across all workers throughout training without any central parameter server.

PyTorch implements this via **DistributedDataParallel (DDP)**. DDP hooks into
the autograd graph and fires a collective communication operation called
**AllReduce** automatically after each backward pass. AllReduce sums the
gradient tensors from all workers, divides by world size, and distributes the
result back to every worker simultaneously. From the training loop's perspective
this is transparent — `loss.backward()` blocks until the AllReduce completes,
then `optimizer.step()` applies identical gradients on every worker.

The collective communication backend in this project is **Gloo**, which
implements AllReduce over TCP sockets. This makes it suitable for CPU-only
training and is the correct choice when NCCL (which requires CUDA) is not
available.

### Training Cadence

The synchronization boundary is the **batch**, not the epoch:

- **Batch**: forward pass → backward pass → AllReduce → optimizer step. One
  gradient update. One AllReduce operation over the network.
- **Epoch**: one full pass through the dataset. If you have 1000 samples and
  batch_size=64, that's ~16 batches per epoch — 16 gradient updates and 16
  AllReduce operations before the data is exhausted once.

If worker1 finishes its backward pass before the master, it enters a wait state
inside the AllReduce call. It has posted its gradients but cannot complete the
reduction until all other ranks have also posted. Once all ranks contribute,
the ring-AllReduce runs, both sides receive averaged gradients, and all
`optimizer.step()` calls proceed. No rank moves to the next batch until every
rank has completed the current one.

---

## PyTorchJob → Gloo Initialization Flow

### 1. Controller sees the CRD

The Kubeflow Training Operator runs as a deployment in the cluster and watches
for `pytorchjobs.kubeflow.org` resources via the API server's watch stream.
When a PyTorchJob manifest is applied, the operator's reconcile loop fires and
translates the spec into concrete Pods — one per replica, with roles assigned.
In this phase: one Pod with role `Master` (rank 0) and one with role `Worker`
(rank 1).

### 2. Env var injection

Before the Pods are scheduled, the Training Operator injects environment
variables into each container spec:

| Variable      | Master Pod              | Worker Pod              |
|---------------|-------------------------|-------------------------|
| `MASTER_ADDR` | Master Pod's DNS name   | Same — master's address |
| `MASTER_PORT` | 23456 (default)         | Same                    |
| `RANK`        | 0                       | 1                       |
| `WORLD_SIZE`  | 2                       | 2                       |
| `LOCAL_RANK`  | 0                       | 0                       |

`MASTER_ADDR` is set to the headless Service DNS name the operator creates
alongside the Pods:
`<jobname>-master-0.<jobname>.<namespace>.svc.cluster.local`

This resolves to the master Pod's IP via CoreDNS, which in a Flannel cluster
means a lookup that ultimately hits the VXLAN overlay.

### 3. Rendezvous

When the training script calls `dist.init_process_group(backend="gloo",
init_method="env://")`, PyTorch reads the injected env vars and performs the
rendezvous:

- Rank 0 binds a TCP socket on `MASTER_PORT` and listens.
- Rank 1 connects to `MASTER_ADDR:MASTER_PORT`.
- They exchange capability handshakes and agree on world size.
- Once all ranks have joined, `init_process_group` returns on all processes
  simultaneously — this is a barrier.

If one Pod is slow to start or DNS is not yet resolving, all other ranks block
here. This is why pods can show `Running` status but appear stuck at startup.

### 4. AllReduce data path (Flannel)

After rendezvous, Gloo owns communication. Every AllReduce during training
travels this path:

```
PyTorch DDP → Gloo AllReduce → TCP socket → Pod veth →
cni0 bridge → flannel.1 VXLAN device → eth1 (192.168.56.x) →
libvirt bridge → eth1 on remote node → flannel.1 → cni0 →
remote Pod veth → Gloo receive buffer → DDP apply gradients
```

The VXLAN encapsulation adds overhead and is a source of per-step latency
jitter. This is the core variable being benchmarked across CNI configurations
in Phase 4.

---

## Configuration

| Parameter     | Value                        |
|---------------|------------------------------|
| world_size    | 2 (1 Master + 1 Worker)      |
| backend       | gloo                         |
| init_method   | env://                       |
| batch_size    | 64                           |
| num_steps     | 100                          |
| model         | Linear(512→256→10), ~134K params |
| gradient size | ~534KB per AllReduce         |
| image         | 192.168.56.1:5000/train:v2   |
| MLflow        | http://192.168.56.11:30500   |
| CNI           | Flannel (VXLAN overlay)      |

---

## Results

| Metric              | Phase 2 (world_size=1) | Phase 3 (world_size=2) | Delta  |
|---------------------|------------------------|------------------------|--------|
| avg_step_time       | 1.90ms                 | 4.51ms                 | +137%  |
| avg_samples_per_sec | 33,724                 | 28,405                 | -16%   |
| break-even          | —                      | 3.80ms                 | ABOVE  |

### Per-step observations

| Steps  | avg step_time | Notes                                      |
|--------|---------------|--------------------------------------------|
| 0      | 23.24ms       | Gloo rendezvous + first AllReduce init     |
| 10–40  | ~3.7ms        | Steady-state, near break-even              |
| 50–90  | 3.97–5.51ms   | Flannel VXLAN jitter, scheduling noise     |

### Interpretation

At this model size and batch size, AllReduce overhead dominates compute time.
Adding a second worker caused a 16% throughput regression — the communication
cost of synchronizing ~534KB of gradients over a VXLAN overlay exceeds the
compute time saved by splitting the batch. The step-time variance (3.7ms–5.51ms)
is characteristic of overlay encapsulation jitter and is a key target for the
CNI comparison in Phase 4.

---

## Next: Phase 4

Phase 4 swaps the CNI plugin while holding all other variables constant —
same model, same batch size, same job manifest, same image. Calico (BGP, no
overlay) runs first, then Cilium (eBPF datapath). Each run produces an MLflow
entry comparable against this Flannel baseline.
