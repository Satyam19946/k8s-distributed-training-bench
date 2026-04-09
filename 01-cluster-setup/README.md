# Phase 1: Cluster Setup

## Overview
Provisions a 3-node Kubernetes cluster on local KVM/QEMU virtual machines using
Vagrant + libvirt. Bootstraps the cluster with kubeadm, installs Flannel as the
baseline CNI plugin, and deploys the Kubeflow Training Operator.

## Cluster Topology

| Node    | Role          | IP             | vCPU | RAM  |
|---------|---------------|----------------|------|------|
| ctrl    | control-plane | 192.168.56.10  | 2    | 2GB  |
| worker1 | worker        | 192.168.56.11  | 2    | 2GB  |
| worker2 | worker        | 192.168.56.12  | 2    | 2GB  |

## Host Environment

| Component      | Details                        |
|----------------|--------------------------------|
| Host OS        | Ubuntu 24.04 LTS               |
| CPU            | AMD Ryzen 5 3600 (12 threads)  |
| RAM            | 16GB                           |
| Hypervisor     | KVM/QEMU via Vagrant + libvirt |
| VM Image       | bento/ubuntu-24.04             |

## Network Architecture

Two virtual networks per VM:

| Network         | Subnet              | Interface | Purpose                        |
|-----------------|---------------------|-----------|--------------------------------|
| default (NAT)   | 192.168.121.0/24    | eth0      | VM management, vagrant SSH     |
| k8s-training    | 192.168.56.0/24     | eth1      | Kubernetes node + pod traffic  |

Pod network CIDR: `10.244.0.0/16` (Flannel, one /24 per node)
Service CIDR: `10.96.0.0/12` (kube-proxy managed, virtual IPs)

kubelet on each node is pinned to `eth1` via `KUBELET_EXTRA_ARGS=--node-ip`.

## Software Versions

| Component               | Version  |
|-------------------------|----------|
| Kubernetes              | v1.31.14 |
| containerd              | v2.2.2   |
| Flannel                 | latest   |
| Kubeflow Training Op.   | v1.8.1   |

## What Was Installed

**On all nodes (via Vagrantfile provisioning):**
- containerd (container runtime, systemd cgroup driver)
- kubeadm, kubelet, kubectl (held at v1.31)
- kernel modules: `overlay`, `br_netfilter`
- sysctl: bridge netfilter + ip_forward enabled
- swap disabled (kubeadm hard requirement)
- conntrack (kubeadm preflight requirement, missing from bento box)

**On ctrl only:**
- kubeadm init → bootstrapped control plane
- Flannel CNI → pod networking, nodes flipped to Ready
- Kubeflow Training Operator → PyTorchJob CRD registered

## Verification

```bash
# All nodes Ready
kubectl get nodes -o wide

# Control plane pods healthy
kubectl get pods -n kube-system

# Flannel running on all nodes
kubectl get pods -n kube-flannel

# PyTorchJob CRD registered
kubectl get crd | grep pytorch
```

## Reproducing This Setup

```bash
# 1. Create the libvirt network (one-time, if not already present)
virsh net-define k8s-training-net.xml
virsh net-start k8s-training
virsh net-autostart k8s-training

# 2. Provision VMs
cd 01-cluster-setup
vagrant up

# 3. Bootstrap control plane (on ctrl)
vagrant ssh ctrl
sudo kubeadm init \
  --apiserver-advertise-address=192.168.56.10 \
  --pod-network-cidr=10.244.0.0/16
mkdir -p $HOME/.kube
sudo cp /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# 4. Install Flannel
kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

# 5. Join workers (on worker1 and worker2)
# Use the join command printed by kubeadm init, or generate a new one:
kubeadm token create --print-join-command

# 6. Install Kubeflow Training Operator
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.8.1"
```
