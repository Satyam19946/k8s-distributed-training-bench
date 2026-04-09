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
- [Completed] Phase 1: Cluster setup and operator install
- [In progress] Phase 2: Baseline single-node PyTorchJob with MLflow tracking
- [ ] Phase 3: Multi-node training, baseline throughput measurement
- [ ] Phase 4: CNI and kernel benchmarking loop
- [ ] Phase 5: Analysis and findings
