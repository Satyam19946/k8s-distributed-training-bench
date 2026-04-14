# k8s-distributed-training-bench

Distributed PyTorch training on Kubernetes with systematic benchmarking of
Gloo collective communication performance across CNI configurations.

## Goal
Set up multi-node distributed training on Kubernetes using the Kubeflow Training
Operator, then benchmark and optimize inter-node communication by tuning CNI
configuration, kernel networking parameters, and topology-aware scheduling.
Measure impact on training throughput (samples/sec) and step time.

## Stack
- Kubernetes (kubeadm, 3-node VM cluster via Vagrant + libvirt)
- Kubeflow Training Operator (PyTorchJob CRD)
- PyTorch DDP + Gloo backend (CPU-based multi-node training)
- CNI comparison: Flannel (baseline) → Calico → Cilium
- Prometheus + Grafana for throughput and network metrics
- MLflow for experiment tracking

## Status
- [x] Phase 1: Cluster setup and operator install
- [x] Phase 2: Baseline single-node PyTorchJob with MLflow tracking
- [ ] Phase 3: Multi-node training, baseline throughput measurement
- [ ] Phase 4: CNI and kernel benchmarking loop
- [ ] Phase 5: Analysis and findings

## Artifacts of Phase 1 - Host Startup Sequence
Order matters on every host reboot — do this before `vagrant up`:

1. Ensure k8s-training libvirt network is active:
   `virsh net-list --all` → if inactive: `virsh net-start k8s-training`
2. Start the container registry:
   `docker compose -f ~/Projects/k8s-distributed-training-bench/registry/docker-compose.yml up -d`
3. Verify registry is up:
   `curl http://192.168.56.1:5000/v2/`
4. Bring VMs up:
   `cd ~/Projects/k8s-distributed-training-bench/01-cluster-setup && vagrant up`
5. Set kubectl context:
   `kubectl config use-context kubernetes-admin@kubernetes`

## Phase 2

### Model Architecture
Intentionally minimal — the goal is to isolate AllReduce overhead, not measure
model performance. Compute time is kept small so communication cost is visible.

Input (512) → Linear(512→256) → ReLU → Linear(256→10) → Output

Parameter count: ~133,898 (~534KB of float32 gradients per AllReduce)

At 1Gbps theoretical link speed, 534KB takes ~4ms to transfer. With TCP
overhead and Flannel VXLAN encapsulation the real cost is higher. Since
Phase 2 compute time is 1.9ms, AllReduce will dominate step time in Phase 3 — making CNI overhead clearly measurable.

### Why This Model Size
- Gradient tensor is large enough to make AllReduce cost visible
- Small enough that compute time stays low (~1.9ms), keeping the
  compute/communication ratio unfavorable — this amplifies CNI differences
- No dataset download required — synthetic random tensors only
- Model size is intentionally not configurable yet — Phase 4 will introduce
  a MODEL_SIZE env var to sweep across small/medium/large architectures

### Baseline Numbers (single-node, world_size=1 -> AllReduce cost = 0, Flannel CNI)
| Metric | Value |
|---|---|
| avg_step_time_ms | 1.9 |
| avg_samples_per_sec | 33724 |
| backend | gloo |
| batch_size | 64 |
| num_steps | 100 |