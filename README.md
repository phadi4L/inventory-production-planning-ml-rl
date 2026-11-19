# Inventory & Production Planning: Optimization, Machine Learning and Reinforcement Learning

A simulation-based research project developed as part of a master’s thesis in Industrial Engineering. It investigates production and inventory decisions for four synthetic product profiles with stochastic demand, batch-size constraints, capacity limits and finite shelf life.

## Approaches

| Script | Method |
| --- | --- |
| `scripts/main_ansatz1.py` | (s,S) inventory policy and priority-based production allocation |
| `scripts/main_ansatz2.py` | Mixed-integer linear programming (PuLP/CBC) with exponential-smoothing forecasts |
| `scripts/main_ansatz3.py` | Random Forest demand forecasting combined with MILP |
| `scripts/main_ansatz4.py` | Rolling-retrained Random Forest combined with MILP |
| `scripts/main_RL.py` | Deep Q-Network (DQN) in PyTorch with a custom inventory environment |

Synthetic demand is generated for 104 weeks. Weeks 1–52 are used as historical data and weeks 53–104 as the evaluation period for the classical and MILP-based approaches.

## Getting started

Requires Python 3.10+ (recommended). In a terminal, from the repository directory:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_demand.py
python scripts/main_ansatz1.py
python scripts/main_ansatz2.py
python scripts/main_ansatz3.py
python scripts/main_ansatz4.py
```

The generator creates `data/demand_104_setup2.xlsx`. The evaluation scripts save workbooks in `results/`.

Run the RL experiment separately:

```bash
python scripts/main_RL.py
```

**Note:** RL training is configured for 2,000 episodes and may take significant time. CUDA is used when available; CPU is also supported.

## Evaluation

The scripts report metrics such as demand fulfillment/service level, shortages, inventory, spoilage and production utilization. The DQN additionally uses a reward function that penalizes shortages, waste, inventory and extreme product-level shortages.

## Important methodological note

The DQN uses its own episode-based synthetic demand generator and reward structure. Its outputs should **not** be interpreted as a strictly controlled head-to-head benchmark against the four other methods without harmonizing evaluation scenarios, cost assumptions and random seeds. This repository presents the original thesis implementations with minimal path portability changes.

## Reproducibility and status

The Python scripts have been syntax-checked, but a full end-to-end reproduction of every experiment has **not yet been verified**. Package versions are currently unpinned. Outputs are generated locally and are excluded from version control.

## Data

All demand data is synthetic; no customer or company datasets are included.

## Background

Master’s thesis: *Comparative Analysis of Traditional and AI-Based Methods for Simultaneous Inventory and Production Planning in the Chemical and Pharmaceutical Industries*.
