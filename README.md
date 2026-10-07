# GSE-VAR

**Gated Self-Explaining Vector Autoregression** (GSE-VAR), with an
exogenous-input extension, **GSE-VARX**. This modular PyTorch implementation
combines state-dependent coefficients with gated Neural Granger Causality
(NGC) regularization.

This code is designed around two reference implementations:

- GVAR: <https://github.com/i6092467/GVAR>
- Neural-GC: <https://github.com/iancovert/Neural-GC>

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
    optimizer="ista",
    log_every=10,
    verbose=1,
)

result = fit_gse_var(data, config)

print(result.causal_strength)
print(result.causal_graph)
```

Example training output:

```text
Epoch    1/200  | loss=1.72 | mse=1.51 | ngc=0.21 | smooth=0 | jacobian=0 | active_edges=25/25 (100.00%)
Epoch   10/200  | loss=1.38 | mse=1.22 | ngc=0.16 | smooth=0 | jacobian=0 | active_edges=18/25 (72.00%)
```

## Sparse Group Lasso

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

The default NGC regularizer is:

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

With `optimizer="ista"`, the smooth prediction objective is optimized first,
then the non-negative sparse-group proximal operator is applied to
`causal_gate`:

1. one-sided elementwise thresholding for lag-level sparsity;
2. group soft-thresholding over all lags for each target-source edge.

This preserves non-negativity, can remove individual lag gates, and can also
remove the entire Granger edge.

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
- `optimizer="adam"` adds the nonsmooth penalty directly to the objective and
  projects the gate onto the non-negative orthant after each update. This is
  easy to optimize but does not guarantee exact zeros.
- `optimizer="ista"` optimizes the smooth objective and then applies the
  proximal operator to `causal_gate`, producing exact zeros in the learned
  causal structure.
- The default causal graph threshold is `1e-8` to avoid treating tiny numerical
  residue as a nonzero edge.

## Training Logs

Training logs are controlled by `verbose` and `log_every`.

- `verbose=0`: no logs.
- `verbose=1`: print epoch, loss, MSE, NGC penalty, smoothness penalty, and
  Jacobian penalty, and active edge count.
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

to the differentiable part of either the Adam or ISTA objective and records
the weighted value in `result.history["jacobian"]`. The default is
`lambda_jacobian=0.0`, so existing training behavior and cost are unchanged.
Jacobian agreement and temporal smoothness constrain different properties and
may be enabled independently or together.

During training, the causal gate is treated as a fixed value only inside the
Jacobian penalty. Both its coefficient-input mask and coefficient-output mask
are detached in that branch. The gate continues to receive gradients from the
prediction and smoothness objectives and continues to receive the configured
NGC/ISTA proximal update. This separation prevents the model from reducing
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
    regularizer="hierarchical_group_lasso",
    lambda_ngc=1e-2,
    lambda_ngc_exog=5e-3,
    lambda_smooth=1e-2,
    lambda_jacobian=10.0,
    coefficient_weight_decay=1e-4,
    optimizer="ista",
    device="cuda",
)

result = fit_gse_varx(endog, exog, config)

print(result.endogenous_graph)  # [endogenous target, endogenous source]
print(result.exogenous_graph)   # [endogenous target, exogenous source]
print(result.exogenous_lags)    # [2, 1, 0]
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
endogenous branch. The following optional fields override them for the
exogenous branch; leaving one as `None` reuses the endogenous value:

- `lambda_ngc_exog`
- `lambda_smooth_exog`
- `lambda_jacobian_exog`
- `sparse_group_lambda_exog`
- `sparse_l1_lambda_exog`
- `exogenous_gate_init`

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

## Visualization

The visualization utilities are intended for the GSE-VAR result returned by
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

$$\hat{x}_{i,t}=\sum_{k=1}^{K}\sum_{j=1}^{p}G_{k,i,j}\Phi_{k,i,j}(G_{k,i,:}\odot x_{t-k})x_{j,t-k}$$

である。実装上の形状は次の通り。

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

$$\hat{A}_{i,j}=\mathbf{1}\left[\|G_{:,i,j}\|_2>\tau\right]$$

で構成する。

ここで、$\tau$ は因果グラフを構成するための閾値であり、デフォルトは `1e-8` である。近接勾配法を最適化手法とする場合、$\tau=0$にしてもスパースなグラフが得られる。

### 6. 目的関数

目的関数は、予測損失、時間方向の平滑化、NGC型正則化からなる。

```math
\mathcal{L}
=
\mathcal{L}_{\mathrm{pred}}
+
\mathcal{R}_{\mathrm{smooth}}
+
\mathcal{R}_{\mathrm{ngc}}
```

予測損失は平均二乗誤差である。

```math
\mathcal{L}_{\mathrm{pred}}
=
\frac{1}{N}
\sum_t
\|x_t-\hat{x}_t\|_2^2
```

`optimizer="ista"` の場合、$\mathcal{R}_{\mathrm{ngc}}$は通常の勾配計算には含めず、勾配ステップ後に `causal_gate` へ近接作用素として適用する。

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

Sparse Group Lassoは、edge全体をまとめて削除する圧力と、個別lagを削除する圧力を同時に与える。

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

### 9. ISTAによる近接更新

`optimizer="adam"` の場合、NGC正則化を損失に足して通常の勾配法で最適化する。ただし、Adamだけではgateが厳密なゼロになりにくい。

`optimizer="ista"` の場合、まずsmooth partに対して通常の勾配ステップを行う。

```math
\mathcal{L}_{\mathrm{smooth-part}}
=
\mathcal{L}_{\mathrm{pred}}
+
\lambda_{\mathrm{smooth}}\mathcal{R}_{\mathrm{smooth}}
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

一方、ISTAによる近接更新では、学習過程の途中でgateそのものが厳密にゼロになる。ゼロになったedgeは、その後のforward計算で予測に使われない。

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
\|g_{i,j}\|_2>\tau
\right]
```

$\hat{A}_{i,j}=1$ なら、変数 $j$ の過去が変数 $i$ の予測に寄与すると判定する。

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
3. Sparse Group Lassoにより、edge単位の削除とlag単位の削除を同時に扱う。
4. Hierarchical Group Lassoにより、ラグ方向の入れ子構造を利用したラグ選択を行う。
5. ISTAの近接更新により、`causal_gate` に厳密なゼロを生成する。
6. Granger因果行列は、状態依存係数の事後的な閾値処理ではなく、gate normに基づいて構成する。
7. 有効係数を見ることで、効果の符号やラグごとの分布も解釈できる。
