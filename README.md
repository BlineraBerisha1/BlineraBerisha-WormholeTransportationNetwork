# Wormhole Transportation Network — ESA SpOC2

**Design and Analysis of Algorithms (DAA) project** | Python, NumPy, Numba, NetworkX

This project solves the **ESA SpOC2 Wormhole Transportation Network** optimization problem. The goal is to find valid routes for **12 spaceships** through a large directed network, synchronize their arrival times, and minimize the cost of the **most expensive ship**.

## 1. The problem

The supplied dataset contains:

- **10,000 nodes** (black holes)
- **1,998,600 directed edges** (wormholes)
- **12 ships**, each with allowed starting nodes and a departure delay
- **One shared destination** (node 105 in this dataset)
- **At most 500 jumps per ship**
- **Arrival-time window of 1.0**

Each edge has a **mean travel time** and a **variance**. Routes must follow real directed edges, start at an allowed origin, and end at the common destination. All ships must arrive within the allowed time window.

The objective is to **minimize the maximum official route cost across the 12 ships**. The official evaluator also accounts for penalties when nodes are visited multiple times. Therefore, simply finding the shortest route for each ship separately is not enough.

## 2. My solution idea

I used a **graph-based optimization approach** that combines constructive heuristics, local search, and official constraint verification.

1. **Represent the network efficiently.** The competitive solver converts the graph into a compact **Compressed Sparse Row (CSR)** representation so outgoing edges can be accessed quickly.
2. **Construct feasible routes.** It builds **variable-length paths**, choosing real directed edges and trying different starting points. A Lagrangian-style score balances low variance with the travel time needed to reach the arrival window.
3. **Coordinate all 12 ships.** Routes are constructed for a shared arrival interval. An occupancy penalty discourages excessive reuse of the same nodes.
4. **Improve the bottleneck.** Since the objective depends on the *worst* ship, the search focuses on expensive routes and uses **Adaptive Large Neighborhood Search (ALNS)** to replace entire routes or individual segments.
5. **Check and save improvements.** A fast internal evaluator rejects weak candidates; promising candidates are then checked with the **original, unmodified** \`wormhole_udp.py\`. Only officially feasible improvements are kept as checkpoints.

The repository also contains a separate, modular **genetic algorithm** implementation with population initialization, selection, crossover, mutation, and repair operators. The final reported result is from the **competitive heuristic/ALNS solver**, not a claim that the genetic algorithm alone achieved that score.

## 3. Project structure

\`\`\`text
WormholeTransportationNetwork/
├── data/
│   ├── database.npz            # Original network and mission data
│   └── example.npy             # Supplied example chromosome
├── solver/
│   ├── competitive.py          # Main competitive route-construction + ALNS search
│   ├── optimize.py             # Local route rewiring and improvement
│   ├── main.py                 # Alternative genetic algorithm entry point
│   ├── config/                 # GA configuration
│   ├── data/                   # Data loading and validation
│   ├── model/                  # Graph, paths and chromosome representation
│   ├── evaluation/             # Fitness and constraint checks
│   ├── genetic/                # Selection, crossover, mutation and GA loop
│   ├── heuristics/             # Initialization and pathfinding
│   ├── repair/                 # Origin, path and time-window repair
│   ├── output/                 # Result and submission formatting
│   └── tests/                  # Automated tests
├── results/                    # Saved best solution, metrics and submission
├── run_competitive.py          # Recommended solver entry point
├── run_config.json             # Genetic algorithm settings
├── verify.py                   # Verification using the official evaluator
├── SpOC2_independent_audit.py  # Separate data, route and cost audit
├── wormhole_udp.py             # Original ESA challenge evaluator
└── requirements.txt
\`\`\`

A solution is stored as a **6,000-integer chromosome**: 12 rows × 500 positions. Each row lists the source nodes of a ship's path; zero values pad unused positions, and the common destination is implicit.

## 4. Installation

Requires Python and the dependencies in \`requirements.txt\` (the repository notes testing with Python 3.13).

\`\`\`bash
git clone https://github.com/BlineraBerisha1/BlineraBerisha-WormholeTransportationNetwork.git
cd BlineraBerisha-WormholeTransportationNetwork

python -m venv .venv
python -m pip install -r requirements.txt
\`\`\`

Optionally activate the virtual environment **before** installing dependencies:

- Windows PowerShell: \`.\.venv\Scripts\Activate.ps1\`
- macOS/Linux: \`source .venv/bin/activate\`

## 5. How to run

Run these commands from the repository root.

**Verify the included best solution:**

\`\`\`bash
python verify.py
\`\`\`

**Run the independent audit:**

\`\`\`bash
python SpOC2_independent_audit.py
\`\`\`

**Run automated tests:**

\`\`\`bash
python -m pytest solver/tests -q
\`\`\`

**Continue optimizing the saved solution (recommended solver):**

\`\`\`bash
python run_competitive.py --minutes 10
\`\`\`

The competitive solver loads the best feasible saved checkpoint, explores new routes and segments, verifies improvements with the official evaluator, and updates the output files. It can be rerun to continue searching. A new run is **not guaranteed** to improve the score.

To check and re-export the current checkpoint without scheduling extra search time:

\`\`\`bash
python -m solver.competitive --minutes 0
\`\`\`

Other available entry points:

\`\`\`bash
# Local two-edge / route-rewiring optimization
python -m solver.optimize --resume results/best_solution.npy --minutes 3

# Separate genetic algorithm implementation
python -m solver.main --config run_config.json
\`\`\`

## 6. Final verified result

The **saved solution** in this repository has the following reported results from the original evaluator:

| Metric | Result |
| --- | ---: |
| **Official fitness (lower is better)** | **309.9264269667** |
| Feasible | **Yes** |
| Origin violations | 0 |
| Invalid-path violations | 0 |
| Arrival-window violation | -0.0015361905 |
| Bottleneck ship | 12 |
| Ships reaching the destination | 12 / 12 |

A negative arrival-window violation means the solution satisfies the timing constraint with a small margin.

Important result files:

- \`results/best_solution.npy\` — saved solution chromosome
- \`results/submission.json\` — submission-format output
- \`results/decision_vector.json\` — solution as JSON integers
- \`results/best_metrics.json\` — official fitness and feasibility metrics
- \`results/verification_report.json\` — per-ship costs, times and path lengths
- \`results/competition_history.json\` — recorded improvements
- \`results/fitness_summary.txt\` — readable final result

The independent audit also checks the supplied challenge files' SHA-256 hashes, origin validity, directed edges, destinations, and reconstructed costs.

## 7. Time and space complexity

Let:

- **V** = number of nodes (10,000)
- **E** = number of edges (1,998,600)
- **S** = number of ships (12)
- **L** = maximum route length (500)
- **d** = maximum outgoing degree of a node
- **I** = number of graph-lookahead iterations
- **K** = number of distinct cached lookahead settings

| Operation | Approximate time complexity |
| --- | --- |
| Build sorted graph / CSR representation | **O(E log E + V)** |
| Compute one lookahead table | **O(I(V + E))** |
| Construct one trial route | **O(L × d)** (worst-case bound) |
| Score a complete candidate with the fast evaluator | **O(SL log E + SL log(SL))** |
| Complete optimization | Depends on the number of trials and the chosen time budget |

**Space complexity:** The graph and route data need **O(V + E + SL)** memory. Cached graph-lookahead tables add **O(KV)**, giving **O(V + E + SL + KV)** overall for the principal stored structures (excluding library overhead and temporary evaluation objects).

The full search does not have a single fixed runtime: it explores multiple candidate routes until its configured time budget expires. These estimates describe the main data structures and operations, rather than a guarantee about end-to-end runtime.

## 8. Summary

The main lesson from this project is that a shortest-path algorithm alone cannot solve a **multi-ship, time-constrained, min-max optimization** problem. The practical solution was to combine efficient graph representation, route construction, coordinated arrival times, bottleneck-focused local search, and repeated checking against the official evaluator.

**Reference:** [ESA SpOC2 challenge description](https://optimise.esa.int/challenge/spoc-2-wormhole-transportation-network/About)
