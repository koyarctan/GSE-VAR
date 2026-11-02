# Fixed reference sources

- GVAR: Marcinkevičs & Vogt (2021), *Interpretable Models for Granger Causality Using Self-explaining Neural Networks*.
  https://arxiv.org/abs/2101.07600
  https://github.com/i6092467/GVAR/tree/e06268f87fbe923580af14b2cab26399191747a6
  `gvar/senn.py`, `epoch.py`, `linear_examples.py` were recovered from the previously stored, fixed-commit notebook source snapshots.
  Compatibility changes: prediction `.squeeze(-1)` retains the batch axis; random coefficient draws use `.item()` for NumPy's scalar-assignment rules. The sampling sequence is unchanged.
  The author's temporal-smoothness training calculation is retained, including its next-time input ordering and non-squared L2 norm. It is not relabelled as the paper's squared-norm formula.
- NGC: Tank et al., *Neural Granger Causality*.
  https://arxiv.org/abs/1802.05842
  https://github.com/iancovert/Neural-GC/tree/c3263d40433aaf3acc94c27a0c1abf9b8e9fcedf
  `ngc/synthetic.py` is the fixed author generator; its license is adjacent. cMLP is the project's tested vectorized implementation.
  NGC full-batch stateless ISTA is called in contiguous 100-epoch blocks to save weights; no optimizer momentum or data reshuffling is reset.

The Neural-GC MIT license is retained next to its source. GVAR files are attributed to the pinned upstream revision; that revision contains no root license file. Compatibility changes are listed above.
