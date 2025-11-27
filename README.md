# Robust Time Series Analysis with Topological Data Analysis and Deep Reinforcement Learning

## Prerequisites
- Python 3.12

## Create `venv`
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
## Install dependencies
```bash
pip install -r requirements.txt
```

## Run the notebooks

Ru the notebooks in the `notebooks` folder in the number and letters order. The notebooks are designed to be run in sequence, and each notebook builds on the previous one in most of the cases. The final notebook is the main one that contains the results of the analysis.

## Data

For this repo, ADAUSDT is the only dataset used. The data is downloaded from Binance using the `binance` package. The data is stored in the `data` folder.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Why Results Differ Between Runs

Deep Q-Networks (DQN) involve numerous stochastic processes. Unlike deterministic algorithms, DRL agents learn through random exploration and sampling, meaning two identical training runs can produce different policies with different performance characteristics.

**This is not a bug — it's a fundamental property of the algorithm.**

---

## Sources of Randomness

### 1. Neural Network Weight Initialization

The DQN policy network initializes with random weights before training begins.

| Impact | Description |
|--------|-------------|
| High | Different starting weights lead to different optimization trajectories |
| Result | Two runs may converge to different local optima |

### 2. Experience Replay Sampling

DQN stores past experiences (state, action, reward, next_state) in a replay buffer and randomly samples mini-batches for training.

| Impact | Description |
|--------|-------------|
| High | Different samples produce different gradient updates |
| Result | Network learns different patterns in different order |

### 3. Exploration Strategy (ε-greedy)

During training, the agent takes random actions with probability ε (epsilon) to explore the environment.

| Impact | Description |
|--------|-------------|
| High | Different random actions lead to different experiences |
| Result | Agent discovers different strategies |

### 4. Parallel Environment Execution

When using `SubprocVecEnv` for parallel training, multiple environments run simultaneously in separate processes.

| Impact | Description |
|--------|-------------|
| Medium | OS-level thread scheduling determines which environment returns first |
| Result | Order of experiences in replay buffer varies |

### 5. GPU Non-Determinism

CUDA operations on GPUs can be non-deterministic for performance optimization.

| Impact | Description |
|--------|-------------|
| Low-Medium | Same operations may produce slightly different results |
| Result | Accumulated floating-point differences affect training |

### 6. Python Hash Randomization

Python 3.3+ randomizes hash values by default for security reasons.

| Impact | Description |
|--------|-------------|
| Low | Dictionary ordering may vary |
| Result | Minor variations in execution order |

---