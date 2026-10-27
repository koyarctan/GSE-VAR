# GSE-VAR

**Gated Self-Explaining Vector Autoregression** (GSE-VAR), with an
exogenous-input extension, **GSE-VARX**. This modular PyTorch implementation
combines state-dependent coefficients with gated Neural Granger Causality
(NGC) regularization.

This code is designed around the following reference implementations and methods:

- GVAR: <https://github.com/i6092467/GVAR>
- Neural-GC: <https://github.com/iancovert/Neural-GC>
- VARX-L: an unpenalized prediction intercept
  (Nicholson, Matteson and Bien, 2017, Appendix A.2):
  <https://doi.org/10.1016/j.ijforecast.2017.01.003>

The main adaptation is a learnable non-negative `causal_gate` with shape
`[lag, target, source]`. For each `(lag, target)` pair, the coefficient network
receives the gate-masked input, and the generated time-varying coefficients are
multiplied by the same gate. NGC-style structured regularization is then applied
to the gate so proximal gradient updates can create exact zeros in the learned
Granger structure.

## Installation

```bash
pip install -e .
```

Visualization utilities use Matplotlib and can be installed with:

```bash
pip install -e .[viz]
```

## Minimal Usage

```python
import numpy as np

from gse_var import GSEVARTrainingConfig, fit_gse_var

data = np.random.randn(300, 5).astype("float32")

config = GSEVARTrainingConfig(
    order=4,
    hidden_layer_size=32,
    num_hidden_layers=2,
    max_epochs=200,
    batch_size=64,
    learning_rate=1e-3,
    coefficient_weight_decay=1e-4,
    regularizer="sparse_group_lasso",
    sparse_group_lambda=1e-2,
    sparse_l1_lambda=1e-3,
    log_every=10,
    verbose=1,
)

result = fit_gse_var(data, config)

print(result.causal_strength)
print(result.causal_graph)
print(result.intercept)  # prediction intercept, one value per target
```

Example training output:

```text
Epoch    1/200  | loss=1.72 | mse=1.51 | ngc=0.21 | smooth=0 | jacobian=0 | active_edges=25/25 (100.00%)
Epoch   10/200  | loss=1.38 | mse=1.22 | ngc=0.16 | smooth=0 | jacobian=0 | active_edges=18/25 (72.00%)
```

## Sparse Group Lasso

Both GSE-VAR and GSE-VARX default to `regularizer="sparse_group_lasso"`
and always include an unpenalized intercept. All lags of each endogenous `(target, source)` pair
form one unweighted group. Self-edges are penalized exactly like other edges.
Penalty strengths are user-supplied; zero strengths do not induce sparsity.

GSE-VARX keeps the existing exogenous grouping: all valid lags for each
`(target, original source)` form one group. Current terms are included when
specified. It does **not** automatically adopt the paper's grouping across
all targets at a fixed exogenous source and lag.

The previous `endogenous_grouping` option has been removed. Delete it from
training configurations; there is no grouping switch. Explicit `group_lasso`
and `hierarchical_group_lasso` retain their definitions.

### Edge-wise SGL

Neural-GC's cMLP implementation supports a `GSGL` penalty, described in the
repository as group sparse group lasso. Its proximal update first applies a
lag-level shrinkage and then applies source-level group shrinkage.

In this package the regularized object is not an input-layer tensor but
`causal_gate`:

```python
causal_gate.shape == [lag, target, source]
```

Because each lag-level gate entry is scalar, the Neural-GC lag-level group
shrinkage becomes elementwise L1 shrinkage. The edge-level term remains group
lasso over the whole lag vector for each target-source pair.

For each edge `source j -> target i`, define:

```math
g_{i,j} = (G_{1,i,j}, \dots, G_{K,i,j})
```

The edge-wise NGC regularizer is:

```math
R_{\mathrm{SGL}}(G)
=
\lambda_{\mathrm{group}}
\sum_{i,j} \|g_{i,j}\|_2
+
\lambda_{\mathrm{l1}}
\sum_{k,i,j} |G_{k,i,j}|
```

`sparse_group_lambda` and `sparse_l1_lambda` control the two penalty strengths
directly. `lambda_ngc` is not used by `sparse_group_lasso`; it is reserved for
`group_lasso` and `hierarchical_group_lasso`.

Coefficient/intercept Adam updates and ordinary gate proximal updates alternate.
The gate block uses the non-negative sparse-group proximal operator:

1. one-sided elementwise thresholding for lag-level sparsity;
2. group soft-thresholding over all lags for each target-source edge.

This preserves non-negativity and exact zeros. An edge is absent when all of
its lag gates are zero.

## Alternating proximal optimization

Adam updates coefficient networks and the unpenalized intercept, followed by a
fixed-step Euclidean proximal update for gates with the updated coefficients.
`learning_rate` is Adam's base rate; `gate_learning_rate` is the proximal step
(or the same rate when `None`). There is no line search, trial loss evaluation, step expansion,
or automatic step reduction. An ordinary batch needs two forward/gradient
evaluations, with no extra forward pass solely for logging.

The objective still includes `coefficient_weight_decay / 2 *
sum(coefficient_parameter**2)`. Both weights and network biases receive this
L2 penalty; gates and the prediction intercept do not. The Jacobian
regularizer still detaches gates. A fixed learning rate does not guarantee
decreasing loss or global convergence; non-finite losses/gradients raise an
error rather than automatically retrying at a smaller step.

History reports pre-update `mse`, `ngc`, `smooth`, `jacobian`, `weight_decay`
and their sum `loss`. It also reports `coefficient_step_size`, `gate_step_size`,
`coefficient_gradient_norm`, and `gate_proximal_gradient_norm`. The latter is
the gate-block displacement divided by the fixed step size, conditional
on the updated coefficients; it is not proof of global optimality. The four
optimizer diagnostics are zero for evaluation-only calls.

### Separate coefficient and gate time scales

The fixed training algorithm uses Adam for coefficient networks and the intercept,
but **ordinary Euclidean proximal gradient for gates**. For example:

```python
config = GSEVARTrainingConfig(
    order=1, hidden_layer_size=8, max_epochs=2000, batch_size=10000,
    learning_rate=.003, gate_learning_rate=1.,
    regularizer="group_lasso", lambda_ngc=.1 / 16,
    coefficient_weight_decay=.0001, shuffle=False,
)
```

The objective, grouping, nonnegative gate and input/output masks are unchanged.
L2 is differentiated exactly once; Adam never updates gate moments. SGL and HGL
remain available. The two-gradient-evaluation budget is unchanged. This is not
FISTA, a line search, or a claim of PALM/global convergence. Adam's logged
coefficient step size is its base learning rate, not each coordinate's actual
displacement. An optional `epoch_callback(epoch, model, metrics)` in
`fit_gse_var` can observe post-update states for external validation selection.

`gate_learning_rate=None` uses the same base rate for both blocks.
DisARM also accepts this optional field for a separate **logit** learning rate;
the development-validated binary setting was coefficient LR .003, logit LR .03,
initial opening probability .5, and `lambda_l0=.03 / 16`. Its old .9 probability
initialization remains the default for reproducibility.

These numeric settings were tested on a 16-variable sparse interacting nonlinear
VAR, **not** certified for Citi Bike. Penalty strengths use a sum over gates;
the example divides a per-target strength by 16 to match globally mean-scaled
prediction loss. Do not copy it to a different dimension as an established
optimum. Research protocol, failed ablations, unused-seed confirmation and
reproduction commands are in `experiments/gate_research/REPORT.md` locally.

## Binary gates with DisARM (separate, opt-in route)

`fit_gse_var_disarm` uses Bernoulli structural gates and Adam for both
coefficients and logits, separate from the continuous proximal route. Its gradient
estimator follows Dong, Mnih and Tucker, **DisARM** (NeurIPS 2020), equations
(6)--(8): <https://arxiv.org/abs/2006.10680>.

```python
from gse_var import GSEVARDisARMTrainingConfig, fit_gse_var_disarm

config = GSEVARDisARMTrainingConfig(
    order=6, hidden_layer_size=8, max_epochs=3000, batch_size=10000,
    learning_rate=.003, gate_learning_rate=.03,
    lambda_l0=1e-3, gate_probability_init=0.9,
    lambda_smooth=0.01, coefficient_weight_decay=0.01,
    shuffle=False, seed=42, device="cuda:2",
)
result = fit_gse_var_disarm(data, config)
print(result.model.gate_probabilities)  # sigmoid(logits), not causal probabilities
print(result.model.causal_gate)         # fixed 0/1 mask used for inference
print(result.causal_graph)              # any selected lag for each target/source
```

The objective is the expected masked prediction loss (plus the configured
coefficient smoothness/Jacobian penalties), the coefficient-network L2 penalty,
and `lambda_l0 * sum(sigmoid(gate_logits))`. The sum counts **lag routes** over
all targets/sources, including self-routes, not lag-integrated graph edges.
This is an expected L0 selection cost, not a Hard Concrete relaxation and not
the existing SGL/group penalty. Gates and the prediction intercept have no L2
decay. Unlike the existing gate-detached local Jacobian update, the DisARM
logit estimator includes any mask-dependent Jacobian penalty in its sampled
objective; that difference applies only to the new route.

Each step evaluates two antithetic **binary** masks at the same parameter
state. Each mask is shared across all time points in the batch and applies to
both coefficient-generator inputs and coefficient outputs. Coefficient
gradients average the two losses; logit gradients use DisARM plus the analytic
gradient of the expected selection cost. A single joint Adam/SGD step follows.
For each target's gates, DisARM uses that target's MSE, absolute smoothness and
Jacobian loss differences, retaining the original `1 / num_vars` weighting.
This removes unrelated targets' loss fluctuations without changing the
expected objective or rescaling `lambda_l0`. Relative smoothness retains its
full scalar difference because its denominator couples all targets; only
the separable terms are localized in that mode. Coefficient/intercept updates
and scalar loss reporting are unchanged at the same parameters and masks.
There is no straight-through gradient, temperature schedule, line search,
trial-step loop, or post-selection refit. With `lambda_jacobian=0`, there are
exactly two coefficient-network forward/backward evaluations per batch.

Prediction, coefficient plots, and checkpoint validation always use the
deterministic mask `gate_logits >= 0` (opening probability >= 0.5). This is a
per-entry threshold, not a joint MAP optimization or causal probability
estimate. The buffer `causal_gate` and logits are saved in the same state dict;
both must be restored with the coefficient parameters. If manually editing
logits, call `model.update_causal_gate_()` before inference. The optional
`epoch_callback(epoch, model, metrics)` runs after each epoch with the refreshed
fixed mask, without restarting either optimizer or sampling RNG. Training
history is a pre-update two-mask Monte Carlo estimate; validation evaluates
the actual fixed-mask model. Unbiased logit gradients do not guarantee an
accurate graph or a good deterministic-mask forecast.

In `experiments/11_GSE_VAR_lambda簡易探索.ipynb`, set `GSE_METHOD="disarm"`
and edit `DISARM_CONFIG`; use `GSE_METHOD="proximal"` and the preserved
`MODEL_CONFIG` for the continuous proximal route. The single training cell keeps the
same data, linear-SGL comparison, 100-epoch checkpoint selection and figures.
It saves no learning results. Restart the kernel once after updating the
embedded implementation. This older quick notebook is not the paper workflow.
The current notebooks 11–13 automate tuning and final evaluation on Remote SSH.

The public GSE configurations no longer accept `optimizer`, `fit_intercept`,
or `causal_threshold`: remove these arguments from callers. Continuous training
always uses coefficient Adam plus gate prox; DisARM always uses Adam. Gated
models always include an intercept and an edge is selected iff any lag gate is
nonzero. This preserves the retained algorithms' numerical updates, not the
deleted SGD/direct-Adam variants. Checkpoints with intercept parameters keep
their parameter keys and shapes. The quick Citi Bike notebook embeds the new
implementation; restart its kernel after updating. The current full notebooks
11–13 use the repository implementation directly and evaluate both GSE variants.
The quick notebook's helper source is `experiments/_citibike_quick_core.py`;
its packed copy is checked against that source by the notebook tests.

## Remote paper experiments (11–13)

Notebook 11 uses seven US Google Trends queries and monthly US-nationality
visitor counts from JNTO (2004–2019). Notebook 12 evaluates Lorenz96 structure
recovery; notebook 13 evaluates structure/sign recovery on the GVAR paper's
linear VAR. Both synthetic experiments retain T=250 and 500.

Copy the project, including `experiments/paper_experiments/` and `data/`, to the
remote server. Install `python -m pip install -e '.[experiments]'` in the CUDA
environment; the notebooks do not reinstall PyTorch. They automate independent
tuning and freeze settings before final evaluation; no manual quick sweep is
required. See `experiments/README.md` for tmux commands and result formats.

Every fit records configuration, source/environment hashes, numerical arrays,
full GSE histories, and 100-epoch model checkpoints. Final evaluation arrays
include all predictions, effective coefficients, Jacobians, gate support and
DisARM probabilities. CSV/JSON/NPZ allow later plots without retraining; an
independent plotting entry point also exports PDF/PNG/SVG and long-form tables.

## Default initialization

With the default `gate_init=1.0`, GSE-VAR and GSE-VARX initially predict the
most recent endogenous observation. The coefficient-network output weights
start at zero; output biases are zero except for the diagonal of the most
recent endogenous lag, which is one. Exogenous coefficients and the prediction
intercept start at zero. Hidden layers retain Xavier initialization. All
parameters remain trainable; self-edges are not protected from regularization.
No initialization option is required. The ungated GVAR baseline is unchanged.

## Independent prediction intercept

Both models predict `intercept + sum(coefficients * inputs)`. The intercept
is a separate, input-independent parameter with one entry per endogenous
target; it is not a coefficient-network bias, an exogenous column, or a gate.
Following VARX-L Appendix A.2, it receives no sparsity, smoothness, Jacobian,
or coefficient-weight-decay penalty. New fitted models start it at zero and
learn it jointly with the networks. This is not a claim that the nonlinear fit
uses the paper's exact linear centering/profile solution at every update.

Initialization does not use response statistics. A supplied model is not
reinitialized, and VARX early stopping restores the intercept with the complete
best-epoch state. `result.intercept` returns a copy of the learned vector in
model-response units. It is excluded from coefficient
arrays, causal graphs, forests, and coefficient/Jacobian comparisons; its
input derivative is zero. Consequently, all-zero gates leave the fitted
constant prediction, not a forced zero prediction.

There is no intercept switch for GSE models. The separate ungated GVAR baseline
retains its original intercept-free architecture. The persistence predictor is
only the starting point, not a fixed forecasting constraint.

## Design Notes

- The regularization target is `causal_gate`, not input or output layer
  weights. The same gate masks the coefficient-network input and the generated
  coefficient, so an exact zero blocks both the direct coefficient path and the
  indirect coefficient-generation path for the corresponding `source -> target`
  relation.
- Coefficient-network weight decay is separated from the gate. This mitigates
  the scale non-identifiability of `raw_coefficient * causal_gate` without
  weakening exact-zero proximal updates.
- `hierarchical_group_lasso` is still available as an optional structured
  penalty. Lag index `0` is the most distant lag, so nested hierarchical groups
  are prefixes: `[t - order]`, `[t - order, t - order + 1]`, ...,
  `[t - order, ..., t - 1]`.
- Coefficients/intercept use Adam; gates use the exact nonnegative proximal
  operator after the coefficient update. There is no optimizer selection.
- Graphs use exact nonzero support, including tiny nonzero gates. There is no
  post-training threshold setting or threshold search.

## Training Logs

Training logs are controlled by `verbose` and `log_every`.

- `verbose=0`: no logs.
- `verbose=1`: print epoch, loss, MSE, NGC penalty, smoothness penalty, and
  Jacobian penalty, L2 penalty, gate step/residual, and active edge count.
- `log_every=10`: print every 10 epochs, plus the first and final epoch.

## Coefficient--Jacobian Agreement

GSE-VAR's effective coefficient and the local prediction Jacobian are not
generally identical because the coefficient generator itself depends on the
lagged input. For target `i`, source `j`, and lag `k`, the diagnostic compares

```math
C_{t,k,i,j}
\quad\text{with}\quad
J_{t,k,i,j}
=
\frac{\partial \widehat{x}_{t,i}}
{\partial x_{t-k,j}}.
```

Evaluate their agreement on lagged predictors as follows:

```python
from gse_var import (
    construct_lagged_dataset,
    evaluate_jacobian_agreement,
)

lagged = construct_lagged_dataset(data, order=config.order)
agreement = evaluate_jacobian_agreement(
    result.model,
    lagged.predictors,
    batch_size=256,
)

print(agreement.mse)
print(agreement.relative_frobenius_error)
print(agreement.cosine_similarity)
print(agreement.active_sign_agreement)
```

`coefficients`, `jacobian`, and `mismatch` all have shape
`[sample, lag, target, source]`, with `mismatch = jacobian - coefficients`.
The `active_*` summaries exclude structurally zero gate entries, preventing
exact-zero paths from inflating the reported agreement.

Agreement can optionally be encouraged during fitting:

```python
config = GSEVARTrainingConfig(
    order=4,
    hidden_layer_size=32,
    lambda_jacobian=1e-3,
)
```

This adds

```math
\mathcal{R}_{\mathrm{Jac}}
=
\lambda_{\mathrm{Jac}}
\operatorname{mean}_{t,k,i,j}
\left(J_{t,k,i,j} - C_{t,k,i,j}\right)^2
```

to the differentiable part of the alternating training objective and records
the weighted value in `result.history["jacobian"]`. The default is
`lambda_jacobian=0.0`, so existing training behavior and cost are unchanged.
Jacobian agreement and temporal smoothness constrain different properties and
may be enabled independently or together.

During training, the causal gate is treated as a fixed value only inside the
Jacobian penalty. Both its coefficient-input mask and coefficient-output mask
are detached in that branch. The gate continues to receive gradients from the
prediction and smoothness objectives and continues to receive the configured
NGC proximal update. This separation prevents the model from reducing
the Jacobian penalty merely by shrinking or deleting a causal edge; the
coefficient generator must instead improve its local coefficient/Jacobian
agreement. The public configuration remains unchanged: set
`lambda_jacobian > 0` to enable this behavior.

## GSE-VARX

`GSEVARX` separates variables that are predicted from variables that are
only used as predictors. The implementation is application-agnostic:

- `endog` has shape `[time, p]` and contains any variables to predict;
- `exog` has shape `[time, q]` and contains any aligned external predictors;
- neither the number nor the names of variables are fixed by the model.

For example, a marketing application may place brand-level quantities or
sales in `endog` and prices, promotions, displays, holidays, or weather in
`exog`, but these names have no special meaning in the code.

```python
from gse_var import (
    GSEVARXTrainingConfig,
    fit_gse_varx,
)

config = GSEVARXTrainingConfig(
    order=5,                    # endogenous lags 5,...,1
    exog_order=2,               # exogenous lags 2,1
    include_current_exog=True,  # also include exogenous lag 0
    hidden_layer_size=100,
    max_epochs=2000,
    batch_size=128,
    learning_rate=1e-2,
    regularizer="sparse_group_lasso",
    sparse_group_lambda=1e-2,
    sparse_l1_lambda=1e-3,
    sparse_group_lambda_exog=5e-3,
    sparse_l1_lambda_exog=1e-3,
    lambda_smooth=1e-2,
    lambda_jacobian=10.0,
    coefficient_weight_decay=1e-4,
    device="cuda",
)

result = fit_gse_varx(endog, exog, config)

print(result.endogenous_graph)  # [endogenous target, endogenous source]
print(result.exogenous_graph)   # [endogenous target, exogenous source]
print(result.exogenous_lags)    # [2, 1, 0]
print(result.intercept)         # [endogenous target]
```

The returned coefficient arrays are:

```python
result.endogenous_coeffs.shape
# [sample, order, endogenous_target, endogenous_source]

result.exogenous_coeffs.shape
# [sample, number_of_exogenous_terms, endogenous_target, exogenous_source]
```

`include_current_exog=True` is suitable when current external predictors are
known at prediction time, such as a planned promotion or posted price. Set it
to `False` to use only past external predictors; in that case `exog_order`
must be at least one. `exog_order=0` with current exogenous inputs enabled uses
only `exog[t]`.

The endogenous and exogenous branches have separate coefficient generators,
gates, strength maps, and graphs. Existing regularization fields configure the
endogenous branch. `regularizer_exog` selects the exogenous gate penalty;
leaving it as `None` uses the endogenous regularizer. The following optional
fields override their endogenous counterparts:

- `lambda_ngc_exog`
- `lambda_smooth_exog`
- `lambda_jacobian_exog`
- `sparse_group_lambda_exog`
- `sparse_l1_lambda_exog`
- `exogenous_gate_init`

For gate penalty weights, `None` inherits only a weight supported by the
selected exogenous regularizer. Group and hierarchical group lasso use
`lambda_ngc_exog`; sparse group lasso uses `sparse_group_lambda_exog` and
`sparse_l1_lambda_exog`. Unused omitted weights resolve to zero, so selecting
a different exogenous penalty does not require explicit zero overrides.
Same-regularizer inheritance is preserved. An explicitly nonzero
`lambda_ngc_exog` with sparse group lasso, or explicitly nonzero sparse-group
weights with another regularizer, still raises a configuration error.
The smoothness, Jacobian, and gate-initialization overrides always inherit
their endogenous values when omitted.

Coefficient/Jacobian diagnostics are also reported separately:

```python
from gse_var import (
    construct_varx_lagged_dataset,
    evaluate_varx_jacobian_agreement,
)

lagged = construct_varx_lagged_dataset(
    endog,
    exog,
    order=config.order,
    exog_order=config.exog_order,
    include_current_exog=config.include_current_exog,
)
agreement = evaluate_varx_jacobian_agreement(
    result.model,
    lagged.endogenous_predictors,
    lagged.exogenous_predictors,
)

print(agreement.endogenous.relative_frobenius_error)
print(agreement.exogenous.relative_frobenius_error)
```

As in GSE-VAR, Jacobian penalties hold the corresponding gates fixed and
update only their coefficient generators. If contemporaneous exogenous inputs
are included, `exogenous_graph` represents selected conditional predictive
effects, not automatically identified intervention effects. In particular,
prices must satisfy the assumed exogeneity or be handled with an appropriate
identification strategy before interpreting coefficients causally.

### Optional early stopping for GSE-VARX

Early stopping is disabled by default. Enable it with `early_stopping=True`
and explicitly supply `validation_data`: either a `VARXLaggedDataset` (such
as the previously prepared `inputs["valid"]`), or a tuple `(endog_valid,
exog_valid)` of raw/already-expanded blocks with the same lag metadata.
Missing validation data raises an error; training MSE is never silently
substituted. No automatic random split or validation-to-training concatenation
is performed. Use chronological train/validation/test splits, training-fitted
scales and the same feature order; test data must stay out of model selection.
For rolling one-step evaluation, validation blocks may include observed past
context before their first validation response.

```python
from dataclasses import replace

config = replace(
    config,
    early_stopping=True,
    early_stopping_patience=200,
    early_stopping_min_delta=1e-5,
    restore_best_weights=True,
)
result = fit_gse_varx(
    prepared["train"]["endog"], prepared["train"]["exog"], config,
    exog_features=features, endog_names=prepared["endog_names_ja"],
    validation_data=inputs["valid"],
)
print(result.best_epoch, result.epochs_trained, result.best_validation_mse)
```

Each epoch monitors pure validation prediction MSE, averaged over all response
elements (not equally weighted batch means). Sparsity, smoothness, Jacobian and
weight-decay penalties are excluded. Validation uses `eval`/`no_grad` and does
not update gates, coefficients, optimizer state, or training gradients.
`early_stopping_patience` defaults to 100 and must be a positive integer.
`early_stopping_min_delta` defaults to zero: a patience reset requires a drop
strictly greater than this absolute MSE amount, relative to the previous
significant improvement. Small improvements can accumulate toward that drop.

With early stopping enabled, `restore_best_weights=True` (the default) restores
the actual minimum-MSE epoch, including both gates, even if `max_epochs` is
reached before patience expires. The minimum is saved even when its improvement
is smaller than `min_delta`; that setting controls stopping patience only.
Snapshots are held in CPU memory, not written to disk. Set
`restore_best_weights=False` to retain the stopping epoch's weights instead.
Coefficients, strengths and graphs are recomputed on the training inputs after
restoration. `history["val_mse"]` retains every evaluated epoch; its final entry
need not correspond to the restored model. The result also reports `best_epoch`
(one-based), `best_validation_mse`, `epochs_trained`, `early_stopped`, and
`stopped_epoch` (`None` if patience was not reached).

Supplying validation data with `early_stopping=False` only logs validation MSE
and its best epoch: it does not shorten training or restore weights. Existing
calls without validation data retain their original training behavior.
These options currently apply to `GSEVARXTrainingConfig` / `fit_gse_varx`, not
the standalone GSE-VAR or baseline trainers.

### Variable-specific exogenous lags and source-level selection

For different lag sets, keep original source/lag identities rather than
interpreting every expanded column as an unrelated variable. The coefficient
generator still receives only valid packed features; no padded variables or
extra gates are learned. There are two input routes:

1. Raw `exog` columns plus `exog_names` and a complete `exog_lags` mapping.
2. Already expanded columns plus an ordered `ExogenousLayout` of
   `ExogenousFeature` objects, one per input column. **Do not shift them again.**

Both routes require `exog_order=0, include_current_exog=True` to describe the
packed representation. Actual physical offsets live in the layout. Positive
lag `l` means `x[t-l]`, zero means current, and negative means a lead. Leads
must be explicitly declared known at prediction time; this does not establish
statistical exogeneity. Preprocessing/scaling must be fitted on training data
and reused unchanged for validation/test.

```python
from gse_var import GSEVARXTrainingConfig, fit_gse_varx

config = GSEVARXTrainingConfig(
    order=5, hidden_layer_size=100,
    exog_order=0, include_current_exog=True,
    regularizer="sparse_group_lasso",
    sparse_group_lambda=0.01, sparse_l1_lambda=0.005,
    regularizer_exog="sparse_group_lasso",
    sparse_group_lambda_exog=0.01, sparse_l1_lambda_exog=0.005,
    coefficient_weight_decay=1e-4,
)
result = fit_gse_varx(
    endog_train, exog_train, config,
    exog_names=["promo", "oil", "payday", "holiday"],
    exog_lags={"promo": [0, 1, 2, 3, 4], "oil": [1, 2, 3, 4, 5],
               "payday": [-2, -1, 0, 1, 2], "holiday": [0]},
    known_future_exog=["payday"],
)
```

The weights above are examples, not validated hyperparameters. Each target
and original source forms its own gate group across **valid** lags. Sparse
group lasso combines source-level and individual-lag selection. Calendar
sources receive the same gate regularization as other sources. Group norms
and hierarchical nested-group norms are unweighted: there is no extra
`sqrt(group size)` multiplier or group-size-weighting option. Hierarchical
penalties still penalize distant lags through overlapping nested groups. The nonnegative
proximal update retains exact zeros. Jacobian penalties continue to
hold both gates fixed. Hierarchical exogenous penalties reject layouts with
leads: a one-sided lag hierarchy should not silently impose an order on
future and past calendar effects.

Raw lead construction requires enough future known-ahead rows. It trims final
responses if those rows are unavailable. Extra trailing `exog` rows are
allowed in this mode; unrelated unused future cells may be NaN. Blocks are
never crossed. For existing expanded data, alignment is already done by the
preprocessing code, so no additional trimming for physical leads is performed.

For the previously prepared Favorita data (`exog_train` has 150 columns), use
its existing metadata **in the same column order**, without rebuilding CSVs:

```python
from gse_var import ExogenousFeature, ExogenousLayout

info = prepared["feature_info"].loc[prepared["exog_names"]]
features = ExogenousLayout(tuple(
    ExogenousFeature(
        source=str(row.source), lag=int(row.lag), kind=row.kind,
        known_ahead=(row.kind == "payday"),  # these leads are known calendar dates
    )
    for row in info.itertuples()
))
result = fit_gse_varx(
    prepared["train"]["endog"], prepared["train"]["exog"], config,
    exog_features=features, endog_names=prepared["endog_names_ja"],
)
```

`ExogenousFeature.label` optionally sets the original source's display name;
use the same label at every lag, not a label containing a lag suffix.

In layout-aware mode, `result.exogenous_coeffs` retains packed shape
`[sample, 1, target, feature]`, while `exogenous_graph` and
`exogenous_strength` are `[target, original_source]`. Source ordering is
`result.exogenous_source_names`; physical lag ordering is
`result.exogenous_lags`. `result.exogenous_gate_by_lag` and
`result.exogenous_graph_by_lag` have `[lag, target, original_source]` shape,
with NaN for unrequested combinations. `exogenous_term_graph` retains the
individual packed feature decisions. `exogenous_coeffs_by_lag` exposes the
corresponding dense coefficient view `[sample, lag, target, original_source]`.
Binary graph aggregation uses the L2
norm over valid gates **before** applying the threshold, not signed coefficient
sums or OR of individually thresholded lags. Those methods coincide only in
the appropriate zero-threshold case. Legacy uniform-lag results are unchanged.

### VARX visualization and input-specific coefficient distributions

The existing plot functions now support `branch="endogenous"` or
`branch="exogenous"`, different target/source counts, and physical leads,
current inputs and variable-specific lags. Gray cells/spans mean **not
requested**, not estimated zero. Gate plots are continuous by default for
compatibility; use `binary=True` for lag-wise 0/1 matrices.

```python
from gse_var import (
    construct_varx_lagged_dataset, causal_gate_matrices,
    plot_causal_gate_by_lag, plot_causal_graph_matrix,
    plot_edge_lag_boxplots, plot_edge_lag_forest,
)

valid_inputs = construct_varx_lagged_dataset(
    prepared["valid"]["endog"], prepared["valid"]["exog"],
    order=config.order, exog_features=features,
)

plot_causal_gate_by_lag(result, branch="exogenous", binary=True)
plot_causal_graph_matrix(result, branch="exogenous")
matrices = causal_gate_matrices(result, branch="exogenous")
print(matrices.by_lag, matrices.aggregated)

# Evaluate only the passed predictors. Neither validation responses nor the
# training coefficient cache are used; model weights are not updated.
plot_edge_lag_boxplots(result, branch="exogenous", data=valid_inputs, top_n=6)
plot_edge_lag_forest(result, valid_inputs, branch="exogenous",
                     dataset_label="valid", quantiles=(0.05, 0.95), top_n=6)
plot_edge_lag_forest(result, valid_inputs, branch="endogenous",
                     dataset_label="valid", edges=[(0, 1)])
```

Repeat with train/test inputs to display their corresponding coefficient
distributions. Forest dots are period means (or `center="median"`), and bars
are empirical quantiles over the provided inputs, **not confidence intervals**.
The forest function requires explicit input data and never falls back to the
training cache. `evaluate_coefficients(result, valid_inputs)` also exposes
these evaluated coefficient arrays without plotting or refitting. Arrays must
use the same training-fitted scales and feature order. Display values are on
the fitted input/output scales, not automatically intervention effects or
original-unit sales effects.

An exogenous graph containing lag 0 or known-ahead leads is labeled an input
dependency graph, not automatically Granger causality. Set `lag_scope="past"`
in `causal_gate_matrices` or `plot_causal_graph_matrix` to restrict aggregation
to positive lags; sources with no past inputs appear as unused. For a VARX
exogenous plot, use `target_names` and `source_names` for separate labels.

## Visualization

The original visualization calls below apply to the GSE-VAR result returned by
`fit_gse_var`. They use the learned `causal_gate` and the gated effective
coefficient tensor `result.coeffs`.

```python
from gse_var import plot_causal_gate_by_lag, plot_edge_lag_boxplots

variable_names = [f"x{i}" for i in range(data.shape[1])]

plot_causal_gate_by_lag(
    result,
    variable_names=variable_names,
    summary="norm",
    percentile=99,
    tick_label_step=2,
    save_path="causal_gate_by_lag.png",
)
```

`plot_causal_gate_by_lag` draws one heatmap per lag from
`model.causal_gate.shape == [lag, target, source]`. The final panel summarizes
the gate over lags with a norm, max, or mean aggregation. This figure is the
most direct view of the learned structural Granger sparsity.
For larger systems, `tick_label_step` can be used to thin axis labels while
keeping the full cell grid.

```python
plot_edge_lag_boxplots(
    result,
    top_n=6,
    exclude_self=True,
    value="signed",
    variable_names=variable_names,
    save_path="edge_lag_boxplots.png",
)
```

`plot_edge_lag_boxplots` selects the strongest edges from
`result.causal_strength` unless explicit `(target, source)` pairs are supplied:

```python
plot_edge_lag_boxplots(
    result,
    edges=[(1, 0), (2, 1)],  # source 0 -> target 1, source 1 -> target 2
    value="absolute",
)
```

The x-axis is lag and the y-axis is the distribution of
`result.coeffs[:, lag, target, source]` across samples. Use `value="signed"` to
inspect effect direction and sign stability; use `value="absolute"` to inspect
effect magnitude.

## Baselines

For validation experiments, the package also exposes reference-style cMLP and
GVAR baselines with the same high-level fit-result interface:

```python
from gse_var import CMLPTrainingConfig, GVARBaselineTrainingConfig
from gse_var import fit_cmlp, fit_gvar

cmlp_result = fit_cmlp(
    data,
    CMLPTrainingConfig(
        order=4,
        hidden_layer_size=32,
        regularizer="sparse_group_lasso",
        sparse_group_lambda=1e-2,
        sparse_l1_lambda=1e-3,
        optimizer="ista",
    ),
)

gvar_result = fit_gvar(
    data,
    GVARBaselineTrainingConfig(
        order=4,
        hidden_layer_size=32,
        lambda_coeff=1e-3,
        lambda_smooth=1e-3,
    ),
)
```

`fit_cmlp` follows the Neural-GC component-wise MLP design: one MLP is trained
per target variable, and Granger structure is read from the first-layer input
weights. Its sparse group lasso uses direct `sparse_group_lambda` and
`sparse_l1_lambda` parameters, matching the GSE-VAR convention.

`fit_gvar` disables the NGC causal gate and trains the GVAR/SENN-style
state-dependent coefficient model directly. Because this baseline has no
proximal gate, its reported graph is thresholded from coefficient strength
rather than exact structural zeros.

### About the Comparison Implementations

`fit_cmlp` follows the official Neural-GC implementation and trains an independent component-wise MLP for each target variable. The Granger structure is read from the input weights of the first layer.

`fit_gvar` is a pure GVAR/SENN-style model that does not use causal gates. Since this model does not produce exact zeros through proximal gradient updates, `causal_graph` should be interpreted as a comparison graph obtained by thresholding coefficient magnitudes with `causal_threshold`.

## Package Layout

- `gse_var.models`: GSE-VAR and GSE-VARX coefficient models with causal gates.
- `gse_var.gvar`: GVAR baseline without NGC causal gates.
- `gse_var.cmlp`: Neural-GC cMLP baseline.
- `gse_var.regularizers`: sparse group lasso, hierarchical group lasso, and
  proximal operators.
- `gse_var.training`: GSE-VAR training loop, logging, and result objects.
- `gse_var.varx`: GSE-VARX training and result objects.
- `gse_var.interpretability`: coefficient/Jacobian agreement diagnostics.
- `gse_var.visualization`: GSE-VAR causal-gate and effective-coefficient
  plots.
- `gse_var.data`: lagged dataset construction.

---

## アルゴリズム: GSE-VAR — Gated Self-Explaining Vector Autoregression

この節では、本実装の考え方を説明する。
本手法は、GVAR の「状態依存的で符号解釈可能な係数行列」と、Neural Granger Causality (NGC) の「構造的スパース性」を、学習可能な `causal_gate` により統合する。

狙いは次の3点である。

1. 非線形・状態依存的な自己回帰ダイナミクスを表現する。
2. Granger 非因果性に対応する変数間関係を、近接勾配法により厳密にゼロ化する。
3. 学習後に、因果構造は `causal_gate` から、効果の大きさと符号は有効係数から解釈する。

### 1. 多変量時系列解析とVAR

$`p`$変量時系列を次で表す。

```math
x_t = (x_{t,1}, \ldots, x_{t,p})^\top \in \mathbb{R}^{p}
```

通常のVARは、過去の値の線形結合により現在の値を予測する。

```math
x_t
=
\sum_{k=1}^{K}
\Phi_k x_{t-k}
+
\varepsilon_t
```

ここで、$K$ は自己回帰次数、$\Phi_k \in \mathbb{R}^{p \times p}$ はラグ $k$ の係数行列である。$\Phi_{k,i,j}$ は「変数 $j$ の $k$ 期前の値が、変数 $i$ の現在値に与える線形効果」と解釈できる。

VARは係数解釈が容易である一方、非線形性や状態依存性を表現しにくい。本研究は、このVAR的な係数解釈を保ちながら、ニューラルネットワークにより非線形・状態依存的な係数を学習する。

### 2. 提案手法の位置づけ

本手法は、次の2つの既存研究を接続する。

1. **GVAR**: 状態依存的な係数行列をニューラルネットワークで生成し、係数の符号や大きさを解釈できる。
2. **Neural Granger Causality (NGC)**: ニューラルネットワークの重みに構造的スパース制約を課し、Granger因果性を推定する。

GVARは係数解釈に優れるが、係数を厳密なゼロとして選択する仕組みは弱い。NGCは厳密なスパース構造を作れるが、通常のニューラルネットワーク重みはVAR係数のようには解釈しにくい。

そこで本実装では、`causal_gate` を係数生成器への入力と、GVARが出力する状態依存係数の両方に適用する。

```python
causal_gate.shape == [lag, target, source]
```

これにより、状態依存的な効果の表現と、静的なGranger構造の選択を分離する。

### 3. 提案手法の全体像

ラグ付き入力を

$$X_t=(x_{t-K},\ldots,x_{t-1})$$

とする。ラグ $k$、ターゲット $i$ ごとにマスク済み入力

$$u_{k,i,t}=G_{k,i,:}\odot x_{t-k}$$

を作り、ターゲット・ラグ別の係数生成器へ渡す。

$$\Phi_{k,i,:}(u_{k,i,t})=f_{k,i}(u_{k,i,t})$$

最終予測は

$$\hat{x}_{i,t}=\nu_i+\sum_{k=1}^{K}\sum_{j=1}^{p}G_{k,i,j}\Phi_{k,i,j}(G_{k,i,:}\odot x_{t-k})x_{j,t-k}$$

である。$\nu_i$ は入力に依存しない独立した予測切片で、常に有効。スパース罰則と `coefficient_weight_decay` の対象に含めず、0から同時学習する。係数生成ネットワーク内部のバイアスとは別のパラメータである。実装上の形状は次の通り。

```text
coeffs.shape      == [batch, lag, target, source]
causal_gate.shape == [lag, target, source]
```

### 4. Causal Gateによる全経路の構造制御

`causal_gate` は次の両方へ作用する。

1. 変数 $x_j$ を直接乗じる有効係数
2. ターゲット $i$ の係数生成器へ入る $x_j$ の入力経路

ターゲット軸をネットワーク内部で混合しないため、係数生成器は各 `(lag, target)` に独立したMLPを持ち、テンソル演算でベクトル化して実行する。選択された親変数同士の状態依存的相互作用は維持される。

ゲートは非負とし、構造の有無と強度を担当する。効果の正負は状態依存係数 $\Phi$ が担当する。

$$G_{k,i,j}\geq 0$$

### 5. 本研究におけるGranger非因果性

$u_{k,i,j}=G_{k,i,j}x_{j,t-k}$ であるため、入力微分は

$$\frac{\partial \hat{x}_{i,t}}{\partial x_{j,t-k}}=G_{k,i,j}\left[\Phi_{k,i,j}+\sum_{\ell=1}^{p}G_{k,i,\ell}x_{\ell,t-k}\frac{\partial\Phi_{k,i,\ell}}{\partial u_{k,i,j}}\right]$$

となる。したがって、

$$G_{k,i,j}=0\quad\Longrightarrow\quad\frac{\partial \hat{x}_{i,t}}{\partial x_{j,t-k}}=0.$$

全ラグで

$$G_{1,i,j}=\cdots=G_{K,i,j}=0$$

なら、$x_j$の過去から $x_i$ の予測への直接経路と係数生成経路がすべて遮断される。

学習後のGranger因果行列は従来どおり

$$\hat{A}_{i,j}=\mathbf{1}\left[\exists k:\ G_{k,i,j}\ne 0\right]$$

で構成する。

全ラグの厳密なゼロだけを非選択とする。因果グラフの閾値を設定・探索するオプションはない。

### 6. 目的関数

目的関数は、予測損失、時間方向の平滑化、NGC型正則化、および指定した場合のヤコビアン正則化・係数ネットワークのL2正則化からなる。

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{pred}}
+
\mathcal{R}_{\mathrm{smooth}}
+
\mathcal{R}_{\mathrm{ngc}}
+\mathcal{R}_{\mathrm{Jac}}
+\frac{\mu}{2}\|\theta_{\mathrm{coef}}\|_2^2
```

予測損失は平均二乗誤差である。

```math
\mathcal{L}_{\mathrm{pred}}
=
\frac{1}{Np}
\sum_t
\|x_t-\hat{x}_t\|_2^2
```

$\mathcal{R}_{\mathrm{ngc}}$は通常の勾配計算には含めず、ゲートの勾配ステップ後に近接作用素として適用する。

### 7. Temporal Smoothness

GVARは時点ごとに係数行列を出力するため、係数が過度に振動する可能性がある。そこで、有効係数の時間変化に平滑化ペナルティを課す。

absolute mode では次を用いる。

```math
\mathcal{R}_{\mathrm{smooth}}
=
\frac{\lambda_{\mathrm{smooth}}}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\|\tilde{\Phi}_{t+1}-\tilde{\Phi}_{t}\|_F^2
```

relative mode では、係数スケールで正規化する。

```math
\mathcal{R}_{\mathrm{smooth}}
=
\frac{1}{|\mathcal{T}|}
\sum_{t\in\mathcal{T}}
\frac{
\|\tilde{\Phi}_{t+1}-\tilde{\Phi}_{t}\|_F^2
}{
\|\tilde{\Phi}_{t}\|_F^2+\varepsilon
}
```

実装ではGVARにのっとり、 `time_index` を用い、隣接時点 $t_{r+1}-t_r=1$ の組にだけ平滑化を課す。

### 8. NGC型正則化

本実装では、`causal_gate` に対してNGC型正則化を適用する。主に次の2種類を使う。

- `sparse_group_lasso`
- `hierarchical_group_lasso`

#### 8.1 Sparse Group Lasso

既定は、予測先・入力元ごとの全ラグを1群にする辺単位SGLである。群サイズ重みは付けず、自己辺と他の辺を同じ方法で正則化する。グループ切替オプションはない。GSE-VARXの外生側も、元変数・予測対象ごとに有効な全ラグをまとめる。

辺単位の Sparse Group Lasso は、edge全体をまとめて削除する圧力と、個別lagを削除する圧力を同時に与える。

edge `j -> i` に対応するラグ方向のgate vectorを次で定義する。

```math
g_{i,j}
=
(G_{1,i,j},\ldots,G_{K,i,j})
```

正則化項は次である。

```math
\mathcal{R}_{\mathrm{SGL}}(G)
=
\lambda_{\mathrm{group}}
\sum_{i=1}^{p}\sum_{j=1}^{p}
\|g_{i,j}\|_2
+
\lambda_{\mathrm{l1}}
\sum_{k=1}^{K}\sum_{i=1}^{p}\sum_{j=1}^{p}
|G_{k,i,j}|
```

実装上、2つの強さは次の設定値で直接指定する。

```python
sparse_group_lambda = lambda_group
sparse_l1_lambda = lambda_l1
```

`sparse_group_lasso` では `lambda_ngc` は使わない。`lambda_ngc` を同時に指定するとエラーにする。

第1項は、あるedge `j -> i` の全ラグをまとめてゼロにする方向に働く。第2項は、edge全体は残しつつ、特定ラグだけをゼロにする方向に働く。

#### 8.2 Hierarchical Group Lasso

Hierarchical Group Lassoは、ラグ方向に入れ子構造を持つgroup lassoである。目的は、不要な遠いラグを優先的に落とし、ラグの自動選択を行うことである。

数式上のラグ $`k`$ は $`x_{t-k}`$ を意味するため、$`k=1`$ が最も近いラグ、$`k=K`$ が最も遠いラグである。この表記では、suffix groupを次で定義する。

```math
\mathcal{G}_{k,i,j}
=
(G_{k,i,j},G_{k+1,i,j},\ldots,G_{K,i,j})
```

正則化項は次である。

```math
\mathcal{R}_{\mathrm{HGL}}(G)
=
\lambda_{\mathrm{ngc}}
\sum_{i=1}^{p}\sum_{j=1}^{p}
\sum_{k=1}^{K}
\|\mathcal{G}_{k,i,j}\|_2
```

一方、実装テンソルは次の順序でラグを持つ。

```python
inputs[:, 0, :]     = x_{t-K}
inputs[:, K - 1, :] = x_{t-1}
```

つまり、テンソル添字では `0` が最も遠いラグである。そのため、実装上のHGL更新はprefix blockに対して行う。

```math
B_{r,i,j}
=
(G^{\mathrm{store}}_{1,i,j},\ldots,G^{\mathrm{store}}_{r,i,j})
```

このprefix更新は、数式上のsuffix group $`\mathcal{G}_{k,i,j}`$ と同じ意味である。ラグの並びが逆に見えるだけなので注意する。

`hierarchical_group_lasso` では、正則化の強さは `lambda_ngc` で指定する。

```python
lambda_ngc = lambda_ngc
```

### 9. 係数Adamとゲート近接勾配の交互更新

各バッチで係数ネットワーク・切片をAdamで更新し、その更新後の係数を使ってゲートを通常の近接勾配で更新する。ゲートにAdamのモーメントは適用しない。以下のsmooth partには、指定したヤコビアン罰則と係数ネットワークのL2罰則も含める。

```math
\mathcal{L}_{\mathrm{smooth-part}}
=
\mathcal{L}_{\mathrm{pred}}
+
\mathcal{R}_{\mathrm{smooth}}
+\mathcal{R}_{\mathrm{Jac}}
+\frac{\mu}{2}\|\theta_{\mathrm{coef}}\|_2^2
```

```math
G^{(m+1/2)}
=
G^{(m)}
-
\eta
\nabla_G
\mathcal{L}_{\mathrm{smooth-part}}
```

その後、非負制約と正則化に対応する近接作用素を `causal_gate` に直接適用する。

```math
G^{(m+1)}
=
\mathrm{prox}_{\eta\mathcal{R}_{\mathrm{ngc}}+\iota_{\{G\geq 0\}}}
\left(G^{(m+1/2)}\right)
```

ここで、$`\iota_{\{G\geq 0\}}`$ は非負制約を表す指示関数である。

係数側のAdam基準学習率は`learning_rate`、ゲート側の固定刻み幅は`gate_learning_rate`（`None`なら前者と共通）で指定する。刻み幅の拡大・半減、ラインサーチ、更新候補の損失再評価は行わない。通常の1バッチは係数用とゲート用のforward・勾配計算各2回だけで、ログ用の追加forwardも行わない。非有限の損失・勾配はエラーにする。

損失履歴は各バッチの更新前の同一状態で記録し、`loss = mse + smooth + jacobian + ngc + weight_decay`となる。`gate_step_size`は固定学習率であり、係数更新後の条件付きゲート近接勾配残差`gate_proximal_gradient_norm`も保存する。ヤコビアン罰則のgate-detachは維持する。

以下では、$`U=G^{(m+1/2)}`$とおく。

#### 9.1 Sparse Group Lassoの非負近接更新

構造ゲートには非負制約を課す。$U=G^{(m+1/2)}$ とすると、まず一方向のL1 thresholdingを適用する。

$$Z_{k,i,j}=\max\left(U_{k,i,j}-\eta\lambda_{\mathrm{l1}},0\right)$$

次に、各edge `j -> i` のラグベクトル

$$z_{i,j}=(Z_{1,i,j},\ldots,Z_{K,i,j}),\qquad n_{i,j}=\|z_{i,j}\|_2$$

へgroup shrinkageを適用する。

$$G^{(m+1)}_{:,i,j}=\begin{cases}0,&n_{i,j}\leq\eta\lambda_{\mathrm{group}},\\[0.6em]\left(1-\dfrac{\eta\lambda_{\mathrm{group}}}{n_{i,j}}\right)z_{i,j},&n_{i,j}>\eta\lambda_{\mathrm{group}}.\end{cases}$$

この更新はゲートの非負性を維持し、個別lagとedge全体の厳密なゼロを生成する。


#### 9.2 Hierarchical Group Lassoの非負近接更新

Hierarchical Group Lassoでは、まずgateを非負領域へ射影し、その後、入れ子になったgroupに対して順番にgroup soft-thresholdingを適用する。

実装の保存順序では、$G^{\mathrm{store}}_{1}$が最も遠いラグ、$G^{\mathrm{store}}_{K}$ が最も近いラグである。$`U=G^{(m+1/2)}`$ とし、初期値を

```math
Z^{(0)}=\max(U,0)
```

とする。$r=1,\ldots,K$ について、各edge `j -> i` のprefix blockを

```math
b_{r,i,j}^{(r-1)}
=
(Z^{(r-1)}_{1,i,j},\ldots,Z^{(r-1)}_{r,i,j})
```

と定義する。そのノルムを

```math
n_{r,i,j}
=
\|b_{r,i,j}^{(r-1)}\|_2
```

とおく。更新は次である。

```math
b_{r,i,j}^{(r)}
=
\begin{cases}
0,
& n_{r,i,j}\leq \eta\lambda_{\mathrm{ngc}},\\[0.6em]
\left(
1-
\dfrac{\eta\lambda_{\mathrm{ngc}}}{n_{r,i,j}}
\right)b_{r,i,j}^{(r-1)},
& n_{r,i,j}>\eta\lambda_{\mathrm{ngc}}.
\end{cases}
```

そして、$\ell \leq r$ の成分をこの $b_{r,i,j}^{(r)}$ で置き換え、$\ell>r$ の成分はそのステップでは変更しない。これを $r=1$から $K$ まで繰り返し、最終的に

```math
G^{(m+1)}=Z^{(K)}
```

とする。

この更新は、数式上のsuffix group

```math
(G_{k,i,j},\ldots,G_{K,i,j})
```

に対するHGL近接更新と対応している。ただし、実装ではラグを古い順に保存しているため、コード上は `param[:lag_idx + 1]` というprefix更新になる。

### 10. 事後的な閾値処理との違い

事後的な閾値処理では、学習中には小さい非ゼロ値のedgeが予測に使われる。その後で係数や重みを閾値以下だからゼロとみなす。

一方、ゲートの近接更新では、学習過程の途中でgateそのものが厳密にゼロになる。ゼロになったedgeは、その後のforward計算で予測に使われない。

したがって、近接勾配法の利点は単に「最後に0/1を決める」ことではなく、構造選択が学習過程に組み込まれる点にある。

### 11. 学習後の解釈

学習後は、主に2つを見る。

1. `causal_gate` から作るGranger因果行列 $\hat{A}$
2. 有効係数 $\tilde{\Phi}_{k,t}$ の大きさと符号

因果行列は次で作る。

```math
\hat{A}_{i,j}
=
\mathbf{1}
\left[
\exists k:\ G_{k,i,j}\ne 0
\right]
```

$\hat{A}_{i,j}=0$ なら、その過去系列から予測への経路が遮断される。非ゼロは経路が残ることを示すだけで、予測への寄与や真の因果性を保証しない。

一方、有効係数

```math
\tilde{\Phi}_{k,t,i,j}
=
G_{k,i,j}\Phi_{k,i,j}\!\left(G_{k,i,:}\odot x_{t-k}\right)
```

を見ることで、どのラグで、どの符号で、どの程度の効果が現れているかを確認できる。



### 12. まとめ

本手法の特徴は次の通りである。

1. GVARにより、非線形・状態依存的な自己回帰係数を学習する。
2. `causal_gate` を係数生成器への入力と生成後の係数に適用し、動的係数と静的なGranger構造を分離する。
3. 辺単位のSGLで各辺の全ラグをまとめて選択し、L1項で個別ラグのゲートも選択する。自己辺も同様に正則化する。
4. Hierarchical Group Lassoにより、ラグ方向の入れ子構造を利用したラグ選択を行う。
5. 係数をAdam、ゲートを固定刻み幅の近接勾配で交互に更新し、厳密なゼロを生成する。
6. Granger因果行列は、状態依存係数の事後的な閾値処理ではなく、ゲートの厳密な非ゼロから構成する。
7. 有効係数を見ることで、効果の符号やラグごとの分布も解釈できる。
8. 独立した無罰則の予測切片を持つため、ゲートが全てゼロでも学習された定数予測を維持する。切片は因果エッジには含めない。
