# 3D Container Packing Algorithm with Load Balancing

![Bin Packing Visualization](packing-container.webp) <!-- Replace with actual image if available -->

## 📦 Overview

This repository implements a three-stage heuristic algorithm to solve the 3D Single Bin-Size Bin Packing Problem (3D-SBSPP) with balancing constraints. The solution optimally packs boxes of varying dimensions and weights into containers while considering:

- Six possible box orientations
- Weight distribution constraints
- Volume optimization
- Minimum number of containers

## 🔧 Algorithm Stages

### Stage 1: Layer Generation
Generates all possible horizontal layers of identical boxes in all 6 orientations

### Stage 2: Solution Candidates
Creates feasible packing combinations from the generated layers

### Stage 3: Container Packing
Packs boxes into container using the best candidates while ensuring:
- Weight limits are respected
- Load is balanced (50% heaviest + 50% lightest items)
- Maximum volume utilization

## 📊 Features

- **Non-oriented packing**: 6 possible box orientations
- **Load balancing**: Optimal weight distribution
- **Real-world constraints**:
  - Container weight limits
  - Box quantity limits
  - Physical dimensions
- **Efficient computation**: Polynomial time complexity

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- NumPy

### Installation
```bash
git clone https://github.com/Magang-Repo-AI-PB-SPIL/3D-pack-container-problem.git
cd 3D-pack-container-problem
pip install -r requirements.txt
run 3d_container_packing_problem.py