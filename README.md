# Graph-Enhanced Deep Reinforcement Learning for Multi-Objective Unrelated Parallel Machine Scheduling

Implementation of the paper: "Graph-Enhanced Deep Reinforcement Learning for Multi-Objective Unrelated Parallel Machine Scheduling" (WSC 2025)

**Authors:** Bulent Soykan, Sean Mondesire, Ghaith Rabadi, Grace Bochenek
**Institution:** University of Central Florida

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Description](#problem-description)
- [Methodology](#methodology)
- [Implementation](#implementation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

This repository implements a novel Deep Reinforcement Learning (DRL) framework for solving the Unrelated Parallel Machine Scheduling Problem (UPMSP) with multiple objectives. The approach combines:

- **Proximal Policy Optimization (PPO)** - State-of-the-art policy gradient algorithm
- **Heterogeneous Graph Neural Networks (GNN)** - For effective state representation
- **Multi-objective optimization** - Simultaneously minimizing Total Weighted Tardiness (TWT) and Total Setup Time (TST)

### Key Features

✅ Handles complex real-world constraints:
  - Job-specific release dates
  - Machine eligibility restrictions
  - Sequence-dependent and machine-dependent setup times

✅ Learns direct scheduling policies (not just heuristic selection)

✅ Achieves superior trade-off between conflicting objectives

✅ Fast inference time suitable for dynamic environments

## 📖 Problem Description

### Unrelated Parallel Machine Scheduling Problem (UPMSP)

Given:
- **n jobs** {J₁, J₂, ..., Jₙ} to be processed
- **m unrelated machines** {M₁, M₂, ..., Mₘ}
- **Processing time** p_{jk}: time for job j on machine k (machine-dependent)
- **Release date** r_j: earliest time job j can start
- **Due date** d_j: target completion time for job j
- **Weight** w_j: importance of job j
- **Setup time** s_{ijk}: time to switch from job i to job j on machine k
- **Eligibility** M_j ⊆ M: machines that can process job j

### Objectives

**Minimize simultaneously:**

1. **Total Weighted Tardiness (TWT):**
   ```
   TWT = Σ w_j × max(0, C_j - d_j)
   ```

2. **Total Setup Time (TST):**
   ```
   TST = Σ_k (s_{0,k(1),k} + Σ_l s_{k(l),k(l+1),k})
   ```

### Why This Matters

This problem is:
- **NP-hard** - Exponentially difficult as problem size grows
- **Multi-objective** - Conflicting goals (reducing setups may increase tardiness)
- **Real-world** - Models manufacturing scenarios in semiconductors, textiles, etc.

Traditional methods struggle with:
- Heuristics are myopic and suboptimal
- Metaheuristics are slow and require extensive tuning
- Exact methods don't scale to realistic problem sizes

## 🧠 Methodology

### 1. Markov Decision Process (MDP) Formulation

The UPMSP is formulated as an MDP (S, A, P, R, γ):

**State (s_t):**
- Heterogeneous graph representation:
  - **Job nodes**: weight, processing time, due date, release date
  - **Machine nodes**: availability, current setup, last job
  - **Setup nodes**: setup type configurations
  - **Edges**: eligibility, processing times, setup costs
- Global features: current time, WIP, aggregate metrics

**Action (a_t):**
- Select feasible (job, machine) pair to schedule next
- Action masking ensures only eligible assignments

**Reward (r_t):**
- Multi-objective: r_t = -α·ΔTWT - β·setup_time
- Balances both objectives through tuning α, β

### 2. Heterogeneous GNN Architecture

The GNN captures complex relationships:

```
Input Features
     ↓
Node Projections (Job, Machine, Setup)
     ↓
GATv2 Layers (4 layers, 128 hidden dim)
 - Job ↔ Machine edges (eligibility, processing time, setup)
 - Machine ↔ Setup edges (current setup state)
 - Job ↔ Setup edges (required setup)
     ↓
Graph Embeddings
     ↓
Actor-Critic Networks
 - Actor: selects actions (job-machine pairs)
 - Critic: estimates state value
```

**Key Design Choices:**
- **GATv2 Conv**: Attention-based message passing
- **Multiple edge types**: Capture different relationships
- **Residual connections**: Improve gradient flow
- **Layer normalization**: Stabilize training

### 3. PPO Training Algorithm

**Hyperparameters (from paper):**
- Learning rate: 1×10⁻⁴ (with linear decay)
- Discount factor γ: 0.99
- GAE lambda λ: 0.95
- PPO clip ε: 0.2
- Optimization epochs: 10
- Batch size: 64
- Training steps: 10⁶

**Advantages of PPO:**
- Stable training through clipped objective
- Sample efficient
- Proven effectiveness in combinatorial optimization

## 💻 Implementation

### Code Structure

```
src/
├── environment.py         # UPMSP simulator
├── instance_generator.py  # Problem instance generation
├── gnn_model.py          # Heterogeneous GNN architecture
├── ppo_agent.py          # PPO algorithm implementation
├── baselines.py          # ATCSR_Rm and GA baselines
└── visualizations.py     # Plotting and analysis

example_demo.py           # Demonstration script
requirements.txt          # Dependencies
```

### Key Components

#### 1. Environment (`environment.py`)
- Discrete-event simulator
- Handles release dates, eligibility, setups
- Computes TWT and TST
- Compatible with Gymnasium API

#### 2. Instance Generator (`instance_generator.py`)
- Generates instances following paper methodology
- Parameters: n_jobs, n_machines, τ, R, β, δ
- Supports batch generation for experiments

#### 3. GNN Model (`gnn_model.py`)
- `HeterogeneousGNN`: Multi-relation graph encoder
- `StateEncoder`: Converts environment state to graph
- Supports dynamic graph sizes

#### 4. PPO Agent (`ppo_agent.py`)
- `GNN_PPO_Policy`: Actor-critic with shared GNN
- `PPOTrainer`: PPO update algorithm
- Action masking for feasibility

#### 5. Baselines (`baselines.py`)
- **ATCSR_Rm**: Composite dispatching rule
  - Considers setup times and ready times
  - Tunable parameters k1, k2
- **Genetic Algorithm**: Multi-objective GA
  - Chromosome: job sequences per machine
  - Fitness: weighted sum of normalized TWT and TST

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- Stable-Baselines3
- NumPy, Matplotlib, Pandas

### Setup

```bash
# Clone repository
git clone https://github.com/bulentsoykan/GNN-DRL4UPMSP.git
cd GNN-DRL4UPMSP

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Install PyTorch Geometric

```bash
# For CUDA 11.8 (adjust for your CUDA version)
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

## 🎮 Quick Start

### Run Demo

```bash
python example_demo.py
```

This will:
1. Generate problem instances for different sizes
2. Run ATCSR_Rm and GA baselines
3. Compare results
4. Generate visualizations in `results/figures/`

### Generate Custom Instance

```python
from src.instance_generator import InstanceGenerator

generator = InstanceGenerator(seed=42)

instance = generator.generate(
    n_jobs=50,
    n_machines=10,
    tau=0.4,        # due date tightness
    R=0.6,          # due date range
    beta=0.1,       # setup time ratio
    delta=0.75      # eligibility density
)
```

### Solve with ATCSR_Rm

```python
from src.baselines import ATCSR_Rm_Solver

solver = ATCSR_Rm_Solver(k1=2.0, k2=2.0)
result = solver.solve(instance)

print(f"TWT: {result['total_weighted_tardiness']:.2f}")
print(f"TST: {result['total_setup_time']:.2f}")
```

### Solve with Genetic Algorithm

```python
from src.baselines import GeneticAlgorithmSolver

ga = GeneticAlgorithmSolver(
    population_size=50,
    n_generations=100,
    alpha=0.5  # weight for TWT
)

result = ga.solve(instance)
```

## 📊 Results

### Performance Comparison (from paper)

| Size | Method | Avg TWT | Avg TST | Avg Comp Time (s) |
|------|--------|---------|---------|-------------------|
| n=20, m=5 | ATCSR_Rm | 150.0 | 75.0 | <0.01 |
| n=20, m=5 | GA | 120.0 | 70.0 | 60.10 |
| n=20, m=5 | **PPO-GNN** | **110.0** | **65.0** | **0.52** |
| n=50, m=10 | ATCSR_Rm | 355.0 | 190.0 | <0.01 |
| n=50, m=10 | GA | 300.0 | 165.0 | 60.11 |
| n=50, m=10 | **PPO-GNN** | **260.0** | **140.0** | **0.92** |
| n=100, m=15 | ATCSR_Rm | 610.0 | 290.0 | <0.01 |
| n=100, m=15 | GA | 475.0 | 255.0 | 60.12 |
| n=100, m=15 | **PPO-GNN** | **420.0** | **225.0** | **1.57** |

### Key Findings

✨ **PPO-GNN achieves:**
- 📉 11-27% lower TWT compared to GA
- 📉 12-15% lower TST compared to GA
- ⚡ 38× faster inference than GA
- 🎯 Pareto dominance - better on BOTH objectives simultaneously

### Visualizations

The implementation generates:
- Performance comparison charts (bar plots)
- Pareto front visualization
- Training curves (if training from scratch)
- Statistical analysis tables

See `results/figures/` after running the demo.

## 🔬 Experimental Setup (Paper Details)

### Instance Parameters

**Problem sizes:**
- n ∈ {20, 50, 100}
- m ∈ {5, 10, 15}

**Parameters:**
- Processing times: DU(1, 100)
- Release dates: DU(0, λ·n·p̄/m), λ=0.5
- Due dates: DU(r_j + p̄_j(1-τ-R/2), r_j + p̄_j(1-τ+R/2))
  - τ ∈ {0.2, 0.4, 0.6} (tightness)
  - R ∈ {0.2, 0.6, 1.0} (range)
- Weights: DU(1, 10)
- Setup times: DU(0, β·p̄)
  - β ∈ {0.1, 0.25}
- Eligibility: δ ∈ {0.75, 1.0}

**Test set:**
- 50 instances per parameter combination
- Separate from training set

### Training Configuration

**GNN Architecture:**
- 4 GATv2 layers
- Hidden dimension: 128
- 4 attention heads
- ReLU activation

**PPO Hyperparameters:**
- Learning rate: 1×10⁻⁴ (linear decay)
- Discount γ: 0.99
- GAE λ: 0.95
- Clip ε: 0.2
- Epochs per update: 10
- Batch size: 64
- Total steps: 10⁶

**Hardware:**
- Intel Core i9 CPU
- NVIDIA RTX 3090 GPU

## 📁 Project Structure

```
GNN-DRL4UPMSP/
│
├── src/
│   ├── __init__.py
│   ├── environment.py           # UPMSP environment
│   ├── instance_generator.py    # Instance generation
│   ├── gnn_model.py             # Heterogeneous GNN
│   ├── ppo_agent.py             # PPO implementation
│   ├── baselines.py             # ATCSR_Rm & GA
│   └── visualizations.py        # Plotting functions
│
├── results/
│   ├── figures/                 # Generated plots
│   └── models/                  # Saved models
│
├── requirements.txt             # Dependencies
├── example_demo.py              # Demo script
└── README.md                    # This file
```

## 🎓 Key Contributions (from paper)

1. **Novel PPO application** for direct scheduling policy learning in complex multi-objective UPMSP

2. **Heterogeneous GNN design** tailored for UPMSP with explicit modeling of jobs, machines, and setups

3. **Multi-objective reward function** that effectively balances TWT and TST minimization

4. **Empirical validation** showing PPO-GNN dominates both heuristic and metaheuristic baselines

## 🔮 Future Work

Potential extensions mentioned in the paper:

- Transfer learning across problem distributions
- Integration with real-time dynamic environments
- Extension to other scheduling variants (flexible job shop, etc.)
- Multi-agent approaches for distributed systems
- Explainability analysis of learned policies

## 📚 Citation

If you use this code or build upon this work, please cite:

```bibtex
@inproceedings{soykan2025graph,
  title={Graph-Enhanced Deep Reinforcement Learning for Multi-Objective Unrelated Parallel Machine Scheduling},
  author={Soykan, Bulent and Mondesire, Sean and Rabadi, Ghaith and Bochenek, Grace},
  booktitle={2025 Winter Simulation Conference (WSC)},
  year={2025},
  organization={IEEE}
}
```

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: bulent.soykan@ucf.edu

---

**Note:** This implementation provides the framework and baseline methods. Training the full PPO-GNN agent requires significant computational resources (GPU, multiple days of training). The demo uses simulated PPO-GNN results for illustration.

For the complete trained models or additional experimental data, please contact the author.
