# Performance Paradoxes in Matching Systems

Numerical companion code for **Performance paradoxes in matching systems are not that rare** (NETGCOOP 2025).

This repository evaluates stability conditions, bottleneck sets, and stationary mean queue lengths in stochastic matching networks. It includes a fixed example comparing a graph with an added-edge variant, and a randomized search for arrival distributions exhibiting a performance paradox.

## Contents

- `simulator.py`: numerical evaluation of the formulas and a fixed graph comparison.
- `testloopmulcore.py`: parallel randomized search for cases where adding an edge increases the mean queue length.
- `CITATION.cff` and `references.bib`: citation metadata.

Despite its filename, `simulator.py` evaluates stationary formulas rather than simulating sample paths.

## Quick start

```bash
git clone https://github.com/shuligraph/matching-paradox-numeric.git
cd matching-paradox-numeric
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python simulator.py
```

The script reports the bottleneck set, its stability margin, and the expected total number of queued items for each stable graph.

For the optional randomized search, install its additional dependencies and run:

```bash
python -m pip install numpy tqdm
python testloopmulcore.py
```

The search currently uses all available CPU cores and stops at the first detected example. Its random seed is not fixed, so results may vary between runs.

## Custom examples

Edit the `# -- Example --` block in `simulator.py` to specify the graph, arrival probabilities `mu`, and the edge to add. Keep the same node labels and arrival probabilities in both graphs, then call `run` for each graph separately.

The computation enumerates independent sets and their permutations, so it is intended for small graphs. Calculations use floating-point arithmetic; comparisons close to the stability boundary require care.

## Reference

**Performance paradoxes in matching systems are not that rare**, NETGCOOP 2025.  
DOI: [10.1007/978-3-032-09315-8_3](https://doi.org/10.1007/978-3-032-09315-8_3).

Please cite the associated paper when using this code in research.
