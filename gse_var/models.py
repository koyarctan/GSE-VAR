from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

AggregationName = Literal["max", "mean", "median"]

# GVARの実装に必要なモジュール
class LagwiseMLP(nn.Module):
    """Independent MLP per lag, evaluated as batched tensor operations.

    This class is retained for compatibility with the existing GVAR baseline.
    The gated GSE-VAR model uses :class:`TargetLagwiseMLP` instead.
    """

    def __init__(
        self,
        num_lags: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        self.num_lags = num_lags
        self.weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(num_lags, dims[layer_idx + 1], dims[layer_idx])
                )
                for layer_idx in range(len(dims) - 1)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.empty(num_lags, dims[layer_idx + 1]))
                for layer_idx in range(len(dims) - 1)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weights, self.biases):
            for lag_idx in range(self.num_lags):
                nn.init.xavier_normal_(weight[lag_idx])
            nn.init.constant_(bias, 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activations = inputs
        final_layer = len(self.weights) - 1
        for layer_idx, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            activations = (
                torch.einsum("bki,koi->bko", activations, weight)
                + bias.unsqueeze(0)
            )
            if layer_idx != final_layer:
                activations = torch.relu(activations)
        return activations

# 提案手法の実装に必要なモジュール
class TargetLagwiseMLP(nn.Module):
    """Independent MLP for every ``(lag, target)`` pair.

    Inputs and outputs have shape ``[batch, lag, target, feature]``.  The
    target axis is never mixed, which is essential when every target uses a
    different structural input mask.
    """

    def __init__(
        self,
        num_lags: int,
        num_targets: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        if num_lags <= 0:
            raise ValueError("num_lags must be positive")
        if num_targets <= 0:
            raise ValueError("num_targets must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        self.num_lags = num_lags
        self.num_targets = num_targets
        self.input_dim = input_dim

        self.weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        num_lags,
                        num_targets,
                        dims[layer_idx + 1],
                        dims[layer_idx],
                    )
                )
                for layer_idx in range(len(dims) - 1)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(num_lags, num_targets, dims[layer_idx + 1])
                )
                for layer_idx in range(len(dims) - 1)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weights, self.biases):
            for lag_idx in range(self.num_lags):
                for target_idx in range(self.num_targets):
                    nn.init.xavier_normal_(weight[lag_idx, target_idx])
            nn.init.constant_(bias, 0.1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                "inputs must have shape [batch, lag, target, feature]"
            )
        expected = (self.num_lags, self.num_targets, self.input_dim)
        if inputs.shape[1:] != expected:
            raise ValueError(
                "expected inputs shape "
                f"[batch, {self.num_lags}, {self.num_targets}, "
                f"{self.input_dim}], got {tuple(inputs.shape)}"
            )

        activations = inputs
        final_layer = len(self.weights) - 1
        for layer_idx, (weight, bias) in enumerate(
            zip(self.weights, self.biases)
        ):
            activations = (
                torch.einsum("bkti,ktoi->bkto", activations, weight)
                + bias.unsqueeze(0)
            )
            if layer_idx != final_layer:
                activations = torch.relu(activations)
        return activations


class GSEVAR(nn.Module):
    """GSE-VAR: gated self-explaining vector autoregression.

    The model produces an effective generalized-coefficient tensor with shape
    ``[batch, lag, target, source]``.  For target ``i`` and lag ``k``, the
    coefficient generator receives only
    ``causal_gate[k, i, :] * inputs[:, k, :]``.  The same non-negative gate is
    then applied to the generated coefficients.

    Consequently, if ``causal_gate[k, i, j] == 0``, source ``j`` has neither a
    direct coefficient path nor an indirect coefficient-generation path to
    target ``i`` at lag ``k``.
    """

    def __init__(
        self,
        num_vars: int,
        order: int,
        hidden_layer_size: int,
        num_hidden_layers: int = 1,
        use_causal_gate: bool = True,
        gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        if num_vars <= 0:
            raise ValueError("num_vars must be positive")
        if order <= 0:
            raise ValueError("order must be positive")
        if hidden_layer_size <= 0:
            raise ValueError("hidden_layer_size must be positive")
        if use_causal_gate and (
            not math.isfinite(gate_init) or gate_init < 0
        ):
            raise ValueError("gate_init must be non-negative")

        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.use_causal_gate = use_causal_gate

        if use_causal_gate:
            # Target-specific coefficient generators are required because each
            # target has a different structural input mask.
            self.coeff_net = TargetLagwiseMLP(
                num_lags=order,
                num_targets=num_vars,
                input_dim=num_vars,
                hidden_dim=hidden_layer_size,
                output_dim=num_vars,
                num_hidden_layers=num_hidden_layers,
            )
        else:
            # Preserve the original ungated GVAR baseline architecture and its
            # checkpoint/parameter shapes.
            self.coeff_net = LagwiseMLP(
                num_lags=order,
                input_dim=num_vars,
                hidden_dim=hidden_layer_size,
                output_dim=num_vars * num_vars,
                num_hidden_layers=num_hidden_layers,
            )

        if use_causal_gate:
            gate = torch.full(
                (order, num_vars, num_vars),
                float(gate_init),
            )
            self.causal_gate = nn.Parameter(gate)
        else:
            self.register_parameter("causal_gate", None)

    def reset_parameters(self) -> None:
        self.coeff_net.reset_parameters()

    @torch.no_grad()
    def project_causal_gate_(self) -> torch.Tensor:
        """Project the structural gate onto the non-negative orthant."""
        if self.causal_gate is None:
            raise RuntimeError(
                "project_causal_gate_ requires use_causal_gate=True"
            )
        self.causal_gate.clamp_(min=0.0)
        return self.causal_gate

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, coeffs, _, _ = self._coefficient_components(inputs)
        preds = torch.einsum("bkij,bkj->bi", coeffs, inputs)
        return preds, coeffs

    def _coefficient_components(
        self,
        inputs: torch.Tensor,
        *,
        require_coefficient_input_grad: bool = False,
        detach_causal_gate: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [batch, lag, variables]")
        if inputs.shape[1:] != (self.order, self.num_vars):
            raise ValueError(
                f"expected inputs shape [batch, {self.order}, "
                f"{self.num_vars}], got {tuple(inputs.shape)}"
            )

        if self.causal_gate is None:
            raw_coeffs = self.coeff_net(inputs).reshape(
                inputs.shape[0],
                self.order,
                self.num_vars,
                self.num_vars,
            )
            coeffs = raw_coeffs
            coefficient_inputs = None
            causal_gate = None
        else:
            causal_gate = (
                self.causal_gate.detach()
                if detach_causal_gate
                else self.causal_gate
            )
            gate = causal_gate.unsqueeze(0)
            target_inputs = inputs.unsqueeze(2)
            coefficient_inputs = target_inputs * gate
            if (
                require_coefficient_input_grad
                and not coefficient_inputs.requires_grad
            ):
                coefficient_inputs.requires_grad_(True)
            raw_coeffs = self.coeff_net(coefficient_inputs)
            coeffs = raw_coeffs * gate
        return raw_coeffs, coeffs, coefficient_inputs, causal_gate

    def _jacobian_components(
        self,
        inputs: torch.Tensor,
        *,
        create_graph: bool,
        detach_causal_gate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.causal_gate is None:
            raise RuntimeError(
                "Jacobian calculation requires use_causal_gate=True"
            )

        with torch.enable_grad():
            raw_coeffs, coeffs, coefficient_inputs, causal_gate = (
                self._coefficient_components(
                    inputs,
                    require_coefficient_input_grad=True,
                    detach_causal_gate=detach_causal_gate,
                )
            )
            if coefficient_inputs is None or causal_gate is None:
                raise RuntimeError(
                    "Jacobian shortcut requires gated coefficient inputs"
                )
            contracted_derivative = torch.autograd.grad(
                outputs=raw_coeffs,
                inputs=coefficient_inputs,
                grad_outputs=coefficient_inputs,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
            mismatch = contracted_derivative * causal_gate.unsqueeze(0)
            jacobian = coeffs + mismatch

        return coeffs, jacobian, mismatch

    def forward_with_jacobian(
        self,
        inputs: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return predictions, coefficients, their Jacobian, and mismatch.

        The coefficient tensor and prediction Jacobian both use the layout
        ``[batch, lag, target, source]``.  The Jacobian is
        ``d prediction[target] / d inputs[lag, source]`` and ``mismatch`` is
        ``jacobian - coefficients``.

        Target/lag independence in :class:`TargetLagwiseMLP` makes it possible
        to compute the mismatch with one vector-Jacobian product rather than
        one reverse pass per target.  ``create_graph=True`` retains the graph
        needed to optimize a penalty on this mismatch.
        """
        coeffs, jacobian, mismatch = self._jacobian_components(
            inputs,
            create_graph=create_graph,
            detach_causal_gate=False,
        )
        preds = torch.einsum("bkij,bkj->bi", coeffs, inputs)
        return preds, coeffs, jacobian, mismatch

    def coefficient_jacobian_mismatch_for_regularization(
        self,
        inputs: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> torch.Tensor:
        """Return ``J - C`` while treating the causal gate as fixed.

        Both uses of the gate -- the coefficient-network input mask and the
        output coefficient mask -- are detached.  The returned tensor keeps
        gradients for coefficient-network parameters but cannot directly
        shrink or remove causal gates.
        """
        _, _, mismatch = self._jacobian_components(
            inputs,
            create_graph=create_graph,
            detach_causal_gate=True,
        )
        return mismatch

    @torch.no_grad()
    def gate_group_norms(self) -> torch.Tensor:
        """Return ``[target, source]`` norms across lag gates."""
        if self.causal_gate is None:
            raise RuntimeError(
                "gate_group_norms requires use_causal_gate=True"
            )
        return torch.linalg.vector_norm(self.causal_gate, ord=2, dim=0)

    @torch.no_grad()
    def causal_graph_from_gate(self, threshold: float = 0.0) -> torch.Tensor:
        """Return a binary Granger graph from structural gate groups."""
        return (self.gate_group_norms() > threshold).to(torch.int64)

    @staticmethod
    def coefficient_strength(
        coeffs: torch.Tensor,
        aggregation: AggregationName = "max",
    ) -> torch.Tensor:
        """Aggregate coefficients into a ``[target, source]`` strength map."""
        abs_coeffs = coeffs.abs()
        if aggregation == "max":
            return abs_coeffs.amax(dim=(0, 1))
        if aggregation == "mean":
            return abs_coeffs.mean(dim=(0, 1))
        if aggregation == "median":
            return abs_coeffs.median(dim=0).values.median(dim=0).values
        raise ValueError(f"unsupported aggregation: {aggregation}")


class GSEVARX(nn.Module):
    """Variable-agnostic GSE-VAR with separate exogenous inputs.

    Endogenous observations are predicted but exogenous observations are not.
    The two additive branches use separate coefficient generators and
    non-negative structural gates:

    ``prediction = endogenous_coefficients * endogenous_lags``
    ``           + exogenous_coefficients * exogenous_terms``.

    Coefficients and gates have layouts ``[batch, lag, target, source]`` and
    ``[lag, target, source]`` respectively. The target dimension is always the
    number of endogenous variables; source dimensions can differ by branch.
    """

    def __init__(
        self,
        num_endogenous: int,
        num_exogenous: int,
        order: int,
        exog_order: int,
        include_current_exog: bool,
        hidden_layer_size: int,
        num_hidden_layers: int = 1,
        gate_init: float = 1.0,
        exogenous_gate_init: float | None = None,
    ) -> None:
        super().__init__()
        if num_endogenous <= 0:
            raise ValueError("num_endogenous must be positive")
        if num_exogenous <= 0:
            raise ValueError("num_exogenous must be positive")
        if order <= 0:
            raise ValueError("order must be positive")
        if exog_order < 0:
            raise ValueError("exog_order must be non-negative")
        if exog_order == 0 and not include_current_exog:
            raise ValueError(
                "include_current_exog must be True when exog_order is zero"
            )
        if hidden_layer_size <= 0:
            raise ValueError("hidden_layer_size must be positive")
        if not math.isfinite(gate_init) or gate_init < 0:
            raise ValueError("gate_init must be non-negative")
        if exogenous_gate_init is None:
            exogenous_gate_init = gate_init
        if (
            not math.isfinite(exogenous_gate_init)
            or exogenous_gate_init < 0
        ):
            raise ValueError("exogenous_gate_init must be non-negative")

        self.num_endogenous = num_endogenous
        self.num_exogenous = num_exogenous
        self.order = order
        self.exog_order = exog_order
        self.include_current_exog = include_current_exog
        self.num_exogenous_terms = exog_order + int(include_current_exog)
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers

        self.endogenous_coeff_net = TargetLagwiseMLP(
            num_lags=order,
            num_targets=num_endogenous,
            input_dim=num_endogenous,
            hidden_dim=hidden_layer_size,
            output_dim=num_endogenous,
            num_hidden_layers=num_hidden_layers,
        )
        self.exogenous_coeff_net = TargetLagwiseMLP(
            num_lags=self.num_exogenous_terms,
            num_targets=num_endogenous,
            input_dim=num_exogenous,
            hidden_dim=hidden_layer_size,
            output_dim=num_exogenous,
            num_hidden_layers=num_hidden_layers,
        )
        self.endogenous_gate = nn.Parameter(
            torch.full(
                (order, num_endogenous, num_endogenous),
                float(gate_init),
            )
        )
        self.exogenous_gate = nn.Parameter(
            torch.full(
                (
                    self.num_exogenous_terms,
                    num_endogenous,
                    num_exogenous,
                ),
                float(exogenous_gate_init),
            )
        )

    def reset_parameters(self) -> None:
        self.endogenous_coeff_net.reset_parameters()
        self.exogenous_coeff_net.reset_parameters()

    def _validate_inputs(
        self,
        endogenous_inputs: torch.Tensor,
        exogenous_inputs: torch.Tensor,
    ) -> None:
        if endogenous_inputs.ndim != 3:
            raise ValueError(
                "endogenous_inputs must have shape [batch, lag, variables]"
            )
        if exogenous_inputs.ndim != 3:
            raise ValueError(
                "exogenous_inputs must have shape [batch, term, variables]"
            )
        if endogenous_inputs.shape != (
            endogenous_inputs.shape[0],
            self.order,
            self.num_endogenous,
        ):
            raise ValueError(
                "expected endogenous_inputs shape "
                f"[batch, {self.order}, {self.num_endogenous}], got "
                f"{tuple(endogenous_inputs.shape)}"
            )
        if exogenous_inputs.shape != (
            exogenous_inputs.shape[0],
            self.num_exogenous_terms,
            self.num_exogenous,
        ):
            raise ValueError(
                "expected exogenous_inputs shape "
                f"[batch, {self.num_exogenous_terms}, {self.num_exogenous}], "
                f"got {tuple(exogenous_inputs.shape)}"
            )
        if endogenous_inputs.shape[0] != exogenous_inputs.shape[0]:
            raise ValueError(
                "endogenous_inputs and exogenous_inputs batch sizes must match"
            )

    @staticmethod
    def _branch_coefficients(
        inputs: torch.Tensor,
        coefficient_network: TargetLagwiseMLP,
        gate_parameter: torch.Tensor,
        *,
        detach_gate: bool,
        require_input_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = gate_parameter.detach() if detach_gate else gate_parameter
        coefficient_inputs = inputs.unsqueeze(2) * gate.unsqueeze(0)
        if require_input_grad and not coefficient_inputs.requires_grad:
            coefficient_inputs.requires_grad_(True)
        raw_coefficients = coefficient_network(coefficient_inputs)
        coefficients = raw_coefficients * gate.unsqueeze(0)
        return raw_coefficients, coefficients, coefficient_inputs, gate

    @staticmethod
    def _branch_jacobian_components(
        inputs: torch.Tensor,
        coefficient_network: TargetLagwiseMLP,
        gate_parameter: torch.Tensor,
        *,
        create_graph: bool,
        detach_gate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            raw, coefficients, coefficient_inputs, gate = (
                GSEVARX._branch_coefficients(
                    inputs,
                    coefficient_network,
                    gate_parameter,
                    detach_gate=detach_gate,
                    require_input_grad=True,
                )
            )
            contracted_derivative = torch.autograd.grad(
                outputs=raw,
                inputs=coefficient_inputs,
                grad_outputs=coefficient_inputs,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
            mismatch = contracted_derivative * gate.unsqueeze(0)
            jacobian = coefficients + mismatch
        return coefficients, jacobian, mismatch

    def forward(
        self,
        endogenous_inputs: torch.Tensor,
        exogenous_inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(endogenous_inputs, exogenous_inputs)
        _, endogenous_coefficients, _, _ = self._branch_coefficients(
            endogenous_inputs,
            self.endogenous_coeff_net,
            self.endogenous_gate,
            detach_gate=False,
            require_input_grad=False,
        )
        _, exogenous_coefficients, _, _ = self._branch_coefficients(
            exogenous_inputs,
            self.exogenous_coeff_net,
            self.exogenous_gate,
            detach_gate=False,
            require_input_grad=False,
        )
        predictions = torch.einsum(
            "bkij,bkj->bi",
            endogenous_coefficients,
            endogenous_inputs,
        ) + torch.einsum(
            "bkij,bkj->bi",
            exogenous_coefficients,
            exogenous_inputs,
        )
        return predictions, endogenous_coefficients, exogenous_coefficients

    def forward_with_jacobians(
        self,
        endogenous_inputs: torch.Tensor,
        exogenous_inputs: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return branch coefficients, local Jacobians, and mismatches."""
        self._validate_inputs(endogenous_inputs, exogenous_inputs)
        endogenous_coefficients, endogenous_jacobian, endogenous_mismatch = (
            self._branch_jacobian_components(
                endogenous_inputs,
                self.endogenous_coeff_net,
                self.endogenous_gate,
                create_graph=create_graph,
                detach_gate=False,
            )
        )
        exogenous_coefficients, exogenous_jacobian, exogenous_mismatch = (
            self._branch_jacobian_components(
                exogenous_inputs,
                self.exogenous_coeff_net,
                self.exogenous_gate,
                create_graph=create_graph,
                detach_gate=False,
            )
        )
        predictions = torch.einsum(
            "bkij,bkj->bi",
            endogenous_coefficients,
            endogenous_inputs,
        ) + torch.einsum(
            "bkij,bkj->bi",
            exogenous_coefficients,
            exogenous_inputs,
        )
        return (
            predictions,
            endogenous_coefficients,
            exogenous_coefficients,
            endogenous_jacobian,
            exogenous_jacobian,
            endogenous_mismatch,
            exogenous_mismatch,
        )

    def coefficient_jacobian_mismatches_for_regularization(
        self,
        endogenous_inputs: torch.Tensor,
        exogenous_inputs: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return branch ``J-C`` tensors with both gates held fixed."""
        self._validate_inputs(endogenous_inputs, exogenous_inputs)
        _, _, endogenous_mismatch = self._branch_jacobian_components(
            endogenous_inputs,
            self.endogenous_coeff_net,
            self.endogenous_gate,
            create_graph=create_graph,
            detach_gate=True,
        )
        _, _, exogenous_mismatch = self._branch_jacobian_components(
            exogenous_inputs,
            self.exogenous_coeff_net,
            self.exogenous_gate,
            create_graph=create_graph,
            detach_gate=True,
        )
        return endogenous_mismatch, exogenous_mismatch

    @torch.no_grad()
    def project_causal_gates_(self) -> None:
        self.endogenous_gate.clamp_(min=0.0)
        self.exogenous_gate.clamp_(min=0.0)

    @staticmethod
    def _gate_group_norms(gate: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(gate, ord=2, dim=0)

    @torch.no_grad()
    def endogenous_graph_from_gate(
        self,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        return (
            self._gate_group_norms(self.endogenous_gate) > threshold
        ).to(torch.int64)

    @torch.no_grad()
    def exogenous_graph_from_gate(
        self,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        return (
            self._gate_group_norms(self.exogenous_gate) > threshold
        ).to(torch.int64)

    @staticmethod
    def coefficient_strength(
        coefficients: torch.Tensor,
        aggregation: AggregationName = "max",
    ) -> torch.Tensor:
        return GSEVAR.coefficient_strength(
            coefficients,
            aggregation=aggregation,
        )


# Legacy aliases also support loading previously pickled full models.
GVARWithNGCGates = GSEVAR
XNeuralVARX = GSEVARX
