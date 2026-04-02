# MATCHING-ABANDONMENT

Code for the paper:  
**Performance Paradox in Matching Models with Abandonment**  
Author: Shu Li

---

This repository includes:

- **Candidate generator**: Find α configurations where merging items increases system congestion.
- **Precise validator**: Check exact values of normalization constant π₀ and expected total number of items under truncation control.
- **CSV export**: High-precision alphas and metrics for further analysis.

---

## 📁 Structure

- `src/generate_candidates.py`: Batch Monte Carlo search over α and δ
- `src/validate_single_case.py`: Plug α into exact formulas and compute bounds
- `data/results.csv`: Output for analysis or visualization

## 🚀 Requirements

```bash
pip install -r requirements.txt
