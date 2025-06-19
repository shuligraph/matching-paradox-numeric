# matching-paradox-sim

Python implementation for the paper  
**“Performance paradoxes in matching systems are not that rare”**  
_A. Busic · J.M. Fourneau · **S. Li** — NETGCOOP 2025_

---

## Contents

| File | Purpose |
|------|---------|
| `simulator.py` | Core algorithm |
| `README.md` | Project documentation (this file) |
| `CITATION.cff` | Citation metadata (YAML) |
| `references.bib` | BibTeX entry for the paper |
| `requirements.txt` | Python dependencies |
| `LICENSE` | MIT License |
| `.gitignore` | Standard Python ignores |

---

## Quick Start

```bash
git clone https://github.com/shuligraph/matching-paradox-sim.git
cd matching-paradox-sim

python -m venv .venv
source .venv/bin/activate          

pip install -r requirements.txt
python simulator.py                
````

### Example output

```text
--- G1 ---
  Bottleneck set: {1, 4}
  Bottleneck: 0.077080
  E[Q] = 7.861968

--- G1_mod1 ---
  Bottleneck set: {1, 4}
  Bottleneck: 0.077080
  E[Q] = 8.179189
```

---

## Using Your Graph

1. **Open** `simulator.py` and scroll to the `# -- Example --` block.

2. **Replace** that block with your graph:

   ```python
   G = nx.Graph()
   G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])            

   mu = {1: 0.15, 2: 0.30, 3: 0.25, 4: 0.10}

   G1_mod1 = G1.copy(); G1_mod1.add_edge(1, 4)  

   ```

3. **Run** the script again:

   ```bash
   python simulator.py
   ```

   The script will print stability, the bottleneck set, and the expected
   queue length $E[G,&alpha;]$.

---

## Contributing

Bug reports, documentation improvements, and additional examples are welcome.
For large changes, please open an issue first.

---

## License

Released under the **MIT License** (see `LICENSE`).

---

## Citation

See `CITATION.cff` for software citation and `references.bib` for the paper’s
BibTeX entry.
