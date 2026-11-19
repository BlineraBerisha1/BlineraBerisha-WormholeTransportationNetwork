
<div align="center">

# Wormhole Transportation Network

### ESA SpOC2 — Design and Analysis of Algorithms

**Optimizing spaceship routes using graph algorithms and heuristic search**

**Best Fitness: `309.92642697`**

</div>

---

## About the Project

This project is based on the **ESA SpOC2 Wormhole Transportation Network Challenge**.

The task is to find efficient routes for 12 spaceships traveling through a large network of black holes connected by wormholes.

Each spaceship must reach the same destination while respecting path and arrival-time constraints.

The main objective is to **minimize the maximum travel cost among all 12 spaceships**.

This is not a simple shortest-path problem because the routes must also satisfy timing requirements and account for penalties caused by repeated node visits.

##  The Network

| Property | Value |
|:---|---:|
| Black holes (nodes) | 10,000 |
| Wormholes (directed edges) | 1,998,600 |
| Spaceships | 12 |
| Common destination | Node 105 |
| Maximum jumps per ship | 500 |
| Arrival window | 1.0 |
| Solution vector | 6,000 integers |

Each wormhole contains two important values:

- **Mean:** Expected travel time.
- **Variance:** Travel uncertainty contributing to the route cost.

A valid solution must respect allowed starting nodes, existing directed connections, the jump limit, and the shared arrival window.

---

## My Approach

I approached the problem using **graph-based heuristics and iterative optimization**.

Rather than searching through every possible combination of routes, I used efficient data structures and improvement strategies to explore promising solutions.

### 1. Graph Representation

The wormhole network is represented as a directed weighted graph.

The competitive solver uses a **Compressed Sparse Row (CSR)** representation, allowing efficient access to outgoing edges without storing a large adjacency matrix.

### 2. Route Construction

Initial candidate routes are constructed using a heuristic that considers:

- Wormhole variance
- Expected travel time
- Allowed starting nodes
- Arrival-time constraints
- Repeated node usage

A Lagrangian-inspired scoring function balances travel time and route cost.

### 3. Coordinating the Spaceships

The routes cannot be optimized independently.

All 12 ships must arrive within the same allowed time window, so the solver constructs and adjusts paths while considering the arrival times of the other ships.

### 4. Improving the Worst Route

The optimization objective depends on the spaceship with the highest cost.

I therefore focused on improving the **bottleneck ship** using:

- Local search
- Segment replacement
- Variable-length route reconstruction
- Adaptive Large Neighborhood Search (ALNS)

These strategies explore alternative paths while attempting to preserve feasibility.

### 5. Verification and Saving

Candidate solutions are first evaluated using a faster internal evaluator.

Promising improvements are then checked against the original ESA evaluation function.

Only officially feasible improvements are saved, allowing optimization to continue from the best existing solution.

---

##  Project Structure

```text
WormholeTransportationNetwork/
│
├── data/
│   ├── database.npz
│   └── example.npy
│
├── solver/
│   ├── competitive.py
│   ├── optimize.py
│   ├── path.py
│   │
│   ├── data/
│   ├── model/
│   ├── evaluation/
│   ├── heuristics/
│   ├── search
│   ├── output/
│   └── tests/
│
├── results/
│   ├── best_solution.npy
│   ├── best_metrics.json
│   ├── submission.json
│   ├── decision_vector.json
│   ├── verification_report.json
│   ├── competition_history.json
│   └── fitness_summary.txt
│
├── run_competitive.py
├── run_config.json
├── verify.py
├── SpOC2_independent_audit.py
├── wormhole_udp.py
├── requirements.txt
└── README.md
```

### Important Files

| File | Purpose |
|:---|:---|
| `run_competitive.py` | Runs the main competitive solver |
| `solver/competitive.py` | Route construction and ALNS optimization |
| `solver/optimize.py` | Local route improvement |
| `verify.py` | Checks a solution using the original evaluator |
| `SpOC2_independent_audit.py` | Independently audits the final solution |
| `results/` | Stores solutions, metrics and submission files |

---

##  Installation & Usage

### 1. Clone the Repository

```powershell
git clone https://github.com/BlineraBerisha1/BlineraBerisha-WormholeTransportationNetwork.git
cd BlineraBerisha-WormholeTransportationNetwork
```

### 2. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

The project uses Python with NumPy, Numba, NetworkX, Matplotlib, Loguru and pytest.

### 3. Verify the Existing Solution

```powershell
python verify.py
```

### 4. Run the Optimization

```powershell
python run_competitive.py --minutes 10
```

The solver starts from an existing feasible checkpoint and attempts to improve it.

The search duration can be changed using the `--minutes` argument.

### 5. Run the Tests

```powershell
python -m pytest solver/tests -q
```

### 6. Independent Audit

```powershell
python SpOC2_independent_audit.py
```

The audit checks the challenge files, graph connections, routes and costs independently.

---

##  Final Results

The final saved solution was evaluated using the original ESA SpOC2 evaluation function.

| Metric | Result |
|:---|---:|
| **Best Fitness** | **309.92642697** |
| Feasible Solution |  Yes |
| Origin Violations | 0 |
| Path Violations | 0 |
| Window Violation | -0.00153619 |
| Bottleneck Ship | 12 |

**A lower fitness value represents a better solution.**

The negative window violation confirms that the arrival-time constraint is satisfied.

The saved solution and verification reports are available in the [`results/`](results/) directory.

---

##  Time & Space Complexity

Let:

- `V` = Number of nodes
- `E` = Number of edges
- `S` = Number of spaceships
- `L` = Maximum route length
- `d` = Maximum outgoing node degree
- `I` = Number of lookahead iterations

### Time Complexity

| Operation | Complexity |
|:---|:---|
| Graph preprocessing | O(E log E + V) |
| Graph lookahead | O(I(V + E)) |
| Constructing one trial route | O(L × d) |
| Candidate fitness evaluation | O(SL log E + SL log(SL)) |

The overall optimization time depends on how many candidate solutions are explored and the selected search duration.

### Space Complexity

The primary data structures require:

**O(V + E + SL)**

Additional cached graph-lookahead tables require memory proportional to the number of cached settings multiplied by `V`.

Using a sparse graph representation avoids the **O(V²)** memory requirement of an adjacency matrix.

---

##  References

- [ESA SpOC2 Challenge](https://optimise.esa.int/challenge/spoc-2-wormhole-transportation-network/About)
- [ESA GECCO Competition](https://www.esa.int/gsp/ACT/projects/gecco-2023-competition/)

---

<div align="center">

**Design and Analysis of Algorithms Project**

Made with Python 

Blinera Berisha

</div>
