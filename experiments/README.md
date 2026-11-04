# Reproducible experiments

The proposed GSE-VAR uses Bernoulli structural gates, expected L0 regularization and DisARM. Each experiment has one executable entry point and one notebook. Shared fitting, metrics, persistence and reference baselines have one implementation.

Install the project from the repository root:

```bash
python -m pip install -e ".[experiments]"
```

| Experiment | Entry point | Notebook | Evaluation |
|---|---|---|---|
| Lorenz96 | `python -m experiments.lorenz96.run --mode full --device cuda:0` | `lorenz96/experiment.ipynb` | Independent pilot selection; F1 at the selected setting; PR/ROC regularization paths |
| Linear VAR | `python -m experiments.linear_var.run --mode full --device cuda:0` | `linear_var/experiment.ipynb` | Independent pilot selection; positive/negative BA and their mean |
| Favorita | `python -m experiments.favorita.run --device cuda:0` | `favorita/experiment.ipynb` | Minimum validation MSE; held-out MSE, MAE and selected routes |
| Lotka–Volterra | `python -m experiments.lotka_volterra.run --devices cuda:0 cuda:1 cuda:2` | `lotka_volterra/experiment.ipynb` | Verified publication data; signed BA; independent pilot selection; full author GVAR grid |

Synthetic experiments support `--mode smoke --device cpu` for pipeline verification with small data and few epochs. Smoke results are separate from full results. Notebooks require `TRAIN=True` to start fitting.

## Remote sessions

Run long remote experiments inside tmux. Activate the intended virtual environment and change to the repository root before starting the session. For example, Lorenz96 on an available GPU0:

```bash
tmux new-session -s lorenz96 -c "$PWD" "'$(command -v python)' -u -m experiments.lorenz96.run --mode full --device cuda:0; exec bash"
```

The absolute Python path preserves the selected virtual environment. Detach with `Ctrl+B`, then `D`; reconnect with `tmux attach -t lorenz96`. The experiment entry points and notebooks do not start tmux automatically. Select only GPUs available for the experiment. The Lotka–Volterra README provides its multi-GPU tmux command.

## Structure

- `shared.py`: numerical metrics, model fitting, checkpoint arrays and run manifests.
- `synthetic.py`: common simulation, independent pilot selection and baseline fitting.
- `lorenz96/`: the Lorenz96 protocol and its exact ten-candidate grids.
- `linear_var/`: signed linear-VAR recovery.
- `favorita/`: calendar alignment, training-only preprocessing and validation selection.
- `lotka_volterra/`: multi-species predator–prey reproduction, a three-GPU work queue and a portable remote bundle; see its README for the two selection protocols.
- `reference/`: fixed upstream source excerpts and licenses for data generation and baseline fitting.
- `analyze.py`: read-only aggregation of completed runs.
- `paper_figures.py`: figure generation from the saved publication archives.

## Results and reproducibility

Runs save their full configuration, source snapshot, environment, generation/training seeds, training histories, checkpoint weights and evaluation arrays. Coefficient axes are `[sample, lag_oldest_first, target, source]`. An edge is selected if any lag gate is one; self-edges are excluded from recovery metrics. Inference uses probability >= 0.5.

F1 uses the setting chosen on independent pilot data. AUPRC/AUROC for GSE-VAR and NGC use all regularization candidates at the selected width. GVAR and linear VAR use their native score thresholds. PR integration is trapezoidal over observed points; average precision is saved separately. Missing endpoints and nonmonotone paths must be inspected rather than silently repaired.

```bash
python -m experiments.analyze results/12_Lorenz96/full_<run_id>
```

Full runs can require substantial GPU time. Reorganizing the source code does not rerun the publication experiments. Newly generated runs have a different source identifier; the publication uses the preserved arrays and weights in `results_lightweight.zip` and the Favorita archive.

## Favorita input data

Place the official Store Sales competition's `train.csv` and `stores.csv` under `data/favorita/raw/`. The code checks the raw files, selects a store using training data only, excludes windows crossing missing days and freezes training weekday means/scales for validation and test. Raw data are not distributed with the code. See `favorita/README.md` for the exact publication protocol.
