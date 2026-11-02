# GSE-VAR

**Gated Self-Explaining Vector Autoregression** learns nonlinear predictive relationships with state-dependent coefficients and binary structural gates. **GSE-VARX** adds a separate branch for external predictors.

Both models use the same learning method: independent Bernoulli gates, expected L0 selection costs, the DisARM logit-gradient estimator, and one joint Adam update per pair of sampled masks.

## Model

For every lag `k` and prediction target `i`, a separate coefficient network receives the masked input. The same binary mask is applied to its output:

```math
v_{ki,t}=z_{ki}\odot x_{t-k},\qquad
c_{ki,t}=z_{ki}\odot f_{ki}(v_{ki,t}),\qquad
\widehat{x}_{i,t}=b_i+\sum_k c_{ki,t}^{\top}x_{t-k}.
```

A zero gate blocks both the direct coefficient term and the input's effect on other coefficients. If all lag routes from source `j` to target `i` are zero, that model's prediction is independent of the source history. The intercept is separate and unpenalized.

The training objective is

```math
\mathbb E_z[\mathrm{MSE}+\gamma\mathcal S(C)]
+\lambda_0\sum_{k,i,j}\sigma(\alpha_{kij})
```

`alpha` contains trainable logits. `sum(sigmoid(alpha))` is the expected number of selected lag routes, including self-routes. `S` penalizes adjacent coefficient changes.

Each update evaluates two antithetic binary masks at unchanged parameters. Ordinary backpropagation averages the coefficient gradients; DisARM estimates logit gradients from loss differences and adds the analytic expected-L0 gradient. Adam then updates both parameter blocks once. Inference is deterministic in both train and eval mode, using `sigmoid(alpha) >= 0.5`; only the trainer explicitly samples masks.

## Installation

Python3.10 or newer:

```bash
python -m pip install -e .
python -m pip install -e ".[viz,experiments]"
```

The core package requires NumPy and PyTorch. Plotting uses Matplotlib. Reproduction additionally uses SciPy, pandas and a Jupyter kernel.

## GSE-VAR

```python
import numpy as np
from gse_var import GSEVARTrainingConfig, fit_gse_var

data = np.random.default_rng(42).normal(size=(300, 5)).astype('float32')
config = GSEVARTrainingConfig(
    order=4, hidden_layer_size=32, num_hidden_layers=1,
    max_epochs=200, batch_size=64,
    learning_rate=1e-3, gate_learning_rate=1e-2,
    lambda_l0=1e-3, gate_probability_init=0.5,
    lambda_smooth=0.01, coefficient_weight_decay=1e-4,
    seed=42,
)
result = fit_gse_var(data, config)
print(result.causal_graph)   # [target, source], any selected lag
print(result.coeffs.shape)  # [sample, lag, target, source]
print(result.intercept)
print(result.model.gate_probabilities)
```

Lag tensors are ordered oldest first: index `order - 1` is lag1. A list of independent continuous blocks can be supplied instead of one array; lag windows and adjacent-time penalties do not cross block boundaries. Inputs must already be on the intended scale.

`epoch_callback(epoch, model, metrics)` receives the refreshed deterministic mask after each epoch and can save checkpoint weights or evaluate a caller-defined validation set.

## GSE-VARX

Endogenous inputs are predicted; external inputs only supply predictors. Both branches have Bernoulli structural gates with independent logits and separate L0 weights.

```python
from gse_var import GSEVARXTrainingConfig, fit_gse_varx

rng = np.random.default_rng(42)
y = rng.normal(size=(300, 3)).astype('float32')
x = rng.normal(size=(300, 2)).astype('float32')
config = GSEVARXTrainingConfig(
    order=4, exog_order=2, include_current_exog=True,
    hidden_layer_size=16, max_epochs=200,
    lambda_l0=1e-3, lambda_l0_exog=2e-3,
    gate_probability_init=0.5, exogenous_gate_probability_init=0.5,
    learning_rate=1e-3, gate_learning_rate=1e-2,
    lambda_smooth=0.01, seed=42,
)
result = fit_gse_varx(y, x, config, endog_names=('A', 'B', 'C'))
print(result.endogenous_graph)  # [target, endogenous source]
print(result.exogenous_graph)   # [target, external source]
print(result.model.endogenous_gate_probabilities)
print(result.model.exogenous_gate_probabilities)
```

`lambda_l0_exog`, `lambda_smooth_exog`, `lambda_jacobian_exog` and `exogenous_gate_probability_init` inherit their endogenous counterpart when set to `None`. They can be overridden separately. Both branches use the same logit learning rate.

### Variable-specific external lags

Identify original sources and physical lags explicitly. Raw external variables can have different lag sets, and known-ahead inputs can have negative lags:

```python
from gse_var import construct_varx_lagged_dataset

dataset = construct_varx_lagged_dataset(
    y, x, order=4,
    exog_names=('promotion', 'calendar'),
    exog_lags={'promotion': [0, 1, 7], 'calendar': [-1, 0]},
    known_future_exog=['calendar'],
)
```

For `fit_gse_varx`, pass the same metadata as keyword arguments. Already-expanded columns can instead be identified by `ExogenousFeature` entries in an `ExogenousLayout`, passed as `exog_features`. Packed tensors have shape `[sample, 1, feature]`. Each physical source/lag feature has one gate per target, and source-level graphs aggregate selected routes. Missing source/lag combinations appear as NaN in dense views. External identities remain attached to the fitted result and coefficient evaluations.

### Validation and early stopping

```python
from dataclasses import replace

config = replace(config, early_stopping=True, early_stopping_patience=20)
result = fit_gse_varx(
    y[:200], x[:200], config,
    validation_data=(y[200:250], x[200:250]),
)
print(result.best_epoch, result.best_validation_mse)
```

Validation can be a preconstructed `VARXLaggedDataset` or a raw `(endog, exog)` tuple with the same lag metadata. Early stopping monitors pure deterministic validation MSE and restores the complete best state, including both logit tensors and binary masks. `early_stopping_min_delta` controls patience resets; the actual minimum-MSE state is retained. With early stopping disabled, an explicit validation set is monitored without stopping or restoration.

Chronological splits, transformations and any lag context preceding validation/test are supplied by the caller. The library does not split or standardize data automatically.

## Coefficients, graphs and interpretation

```python
from gse_var import evaluate_coefficients, evaluate_jacobian_agreement

dataset = construct_varx_lagged_dataset(
    y, x, order=config.order, exog_order=config.exog_order,
    include_current_exog=config.include_current_exog,
)
evaluation = evaluate_coefficients(result, dataset)
```

`evaluate_coefficients` evaluates only the supplied predictors, never refits or reuses stored training coefficients. Use `evaluate_jacobian_agreement` for VAR and `evaluate_varx_jacobian_agreement` for VARX to compare effective coefficients with local prediction derivatives.

An effective coefficient is a prediction-equation weight. Its sign can differ from the derivative because the coefficient itself depends on the inputs. Structural graphs describe model input dependence conditional on the supplied variable set.

## Visualization

```python
from gse_var import plot_causal_gate_by_lag, plot_edge_lag_forest

plot_causal_gate_by_lag(result, branch='exogenous', binary=True)
plot_edge_lag_forest(result, dataset, branch='exogenous', edges=[(0, 0)])
```

VARX plots support `branch='endogenous'` and `branch='exogenous'`. Packed layouts preserve source names and actual lag/current/lead labels. Coefficient plots require explicit predictors so the displayed distribution corresponds to the requested train/validation/test interval.

## Reproducible experiments

See [experiments/README.md](experiments/README.md). The repository provides one entry point and one notebook per experiment:

| Experiment | Purpose | Entry point |
|---|---|---|
| Lorenz96 | Nonlinear structural recovery | `python -m experiments.lorenz96.run --mode full --device cuda:0` |
| Linear VAR | Positive/negative sign recovery | `python -m experiments.linear_var.run --mode full --device cuda:0` |
| Favorita | Category-sales structure and held-out prediction | `python -m experiments.favorita.run --device cuda:0` |

Synthetic pipelines support `--mode smoke --device cpu`. Full protocols record all seeds, tuning/evaluation separation, normalized L0 grids and source/environment identifiers. Original datasets and result archives are stored separately from the public source code.

NGC's cMLP and the ungated GVAR remain comparison methods. Proximal input-weight updates and signed-weight regularizers belong to these baselines; GSE-VAR and GSE-VARX train binary gates with DisARM.

## Tests

```bash
python -m pytest -q
```

Tests cover exact-enumeration checks of the DisARM estimator, joint VAR/VARX updates, distinct branch costs, input-path exclusion, Jacobians, packed source/lag alignment, checkpoint restoration, deterministic inference and experiment metric definitions.

## References

- Dong, Z., Mnih, A., & Tucker, G. (2020). DisARM: An Antithetic Gradient Estimator for Binary Latent Variables. https://arxiv.org/abs/2006.10680
- Marcinkevičs, R., & Vogt, J. E. (2021). Interpretable Models for Granger Causality Using Self-explaining Neural Networks. https://arxiv.org/abs/2101.07600
- Tank, A., Covert, I., Foti, N., Shojaie, A., & Fox, E. B. (2022). Neural Granger Causality. https://arxiv.org/abs/1802.05842

Fixed source revisions and third-party licenses for reproduction are listed in [experiments/reference/README.md](experiments/reference/README.md).

## Manuscript

The manuscript, Appendix A and figure assets are in [thesis/main.tex](thesis/main.tex). Build the PDF with LuaLaTeX and BibTeX:

```bash
cd thesis
latexmk -lualatex -outdir=build main.tex
```

The compiled manuscript is also available as [thesis/main.pdf](thesis/main.pdf).
