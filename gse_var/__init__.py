"""GSE-VAR: Gated Self-Explaining Vector Autoregression."""

from importlib import import_module

from .data import (
    LaggedDataset,
    VARXLaggedDataset,
    construct_lagged_dataset,
    construct_varx_lagged_dataset,
)

__all__ = [
    "CMLP",
    "CMLPFitResult",
    "CMLPTrainingConfig",
    "GSEVARFitResult",
    "GVARBaselineFitResult",
    "GVARBaselineTrainingConfig",
    "GSEVARTrainingConfig",
    "GSEVAR",
    "JacobianAgreementResult",
    "LaggedDataset",
    "NGCRegularizer",
    "RegularizerName",
    "VARXLaggedDataset",
    "GSEVARX",
    "GSEVARXFitResult",
    "GSEVARXJacobianAgreementResult",
    "GSEVARXTrainingConfig",
    "construct_lagged_dataset",
    "construct_varx_lagged_dataset",
    "evaluate_jacobian_agreement",
    "evaluate_varx_jacobian_agreement",
    "fit_cmlp",
    "fit_gvar",
    "fit_gse_var",
    "fit_gse_varx",
    "group_lasso_penalty",
    "hierarchical_group_lasso_penalty",
    "plot_causal_gate_by_lag",
    "plot_edge_lag_boxplots",
    "prox_lasso_",
    "prox_group_lasso_",
    "prox_hierarchical_group_lasso_",
    "prox_sparse_group_lasso_",
    "sparse_group_lasso_penalty",
]

_LAZY_ATTRS = {
    "CMLP": ("gse_var.cmlp", "CMLP"),
    "CMLPFitResult": ("gse_var.cmlp", "CMLPFitResult"),
    "CMLPTrainingConfig": ("gse_var.cmlp", "CMLPTrainingConfig"),
    "GSEVARFitResult": ("gse_var.training", "GSEVARFitResult"),
    "GVARBaselineFitResult": ("gse_var.gvar", "GVARBaselineFitResult"),
    "GVARBaselineTrainingConfig": ("gse_var.gvar", "GVARBaselineTrainingConfig"),
    "GSEVARTrainingConfig": ("gse_var.training", "GSEVARTrainingConfig"),
    "GSEVAR": ("gse_var.models", "GSEVAR"),
    "JacobianAgreementResult": (
        "gse_var.interpretability",
        "JacobianAgreementResult",
    ),
    "GSEVARX": ("gse_var.models", "GSEVARX"),
    "GSEVARXFitResult": (
        "gse_var.varx",
        "GSEVARXFitResult",
    ),
    "GSEVARXJacobianAgreementResult": (
        "gse_var.interpretability",
        "GSEVARXJacobianAgreementResult",
    ),
    "GSEVARXTrainingConfig": (
        "gse_var.varx",
        "GSEVARXTrainingConfig",
    ),
    "NGCRegularizer": ("gse_var.regularizers", "NGCRegularizer"),
    "RegularizerName": ("gse_var.regularizers", "RegularizerName"),
    "fit_cmlp": ("gse_var.cmlp", "fit_cmlp"),
    "evaluate_jacobian_agreement": (
        "gse_var.interpretability",
        "evaluate_jacobian_agreement",
    ),
    "evaluate_varx_jacobian_agreement": (
        "gse_var.interpretability",
        "evaluate_varx_jacobian_agreement",
    ),
    "fit_gvar": ("gse_var.gvar", "fit_gvar"),
    "fit_gse_var": ("gse_var.training", "fit_gse_var"),
    "fit_gse_varx": ("gse_var.varx", "fit_gse_varx"),
    "group_lasso_penalty": ("gse_var.regularizers", "group_lasso_penalty"),
    "hierarchical_group_lasso_penalty": (
        "gse_var.regularizers",
        "hierarchical_group_lasso_penalty",
    ),
    "plot_causal_gate_by_lag": ("gse_var.visualization", "plot_causal_gate_by_lag"),
    "plot_edge_lag_boxplots": ("gse_var.visualization", "plot_edge_lag_boxplots"),
    "prox_lasso_": ("gse_var.regularizers", "prox_lasso_"),
    "prox_group_lasso_": ("gse_var.regularizers", "prox_group_lasso_"),
    "prox_hierarchical_group_lasso_": (
        "gse_var.regularizers",
        "prox_hierarchical_group_lasso_",
    ),
    "prox_sparse_group_lasso_": ("gse_var.regularizers", "prox_sparse_group_lasso_"),
    "sparse_group_lasso_penalty": ("gse_var.regularizers", "sparse_group_lasso_penalty"),
}


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
