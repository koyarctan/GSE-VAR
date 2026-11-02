import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

def run_epoch(epoch_num: int, model: nn.Module, optimiser: optim, predictors: np.ndarray, responses: np.ndarray,
              time_idx: np.ndarray, criterion: torch.nn.modules.loss, lmbd: float, gamma: float, batch_size: int,
              device: torch.device, alpha=0.5, verbose=True, train=True):
    """
    Runs one epoch through the dataset.

    @param epoch_num: number of the epoch (for bookkeeping only).
    @param model: model.
    @param optimiser: Torch optimiser.
    @param predictors: numpy array with predictor values of shape [N x K x p].
    @param responses: numpy array with response values of shape [N x p].
    @param time_idx: time indices of observations of shape [N].
    @param criterion: base loss criterion (e.g. MSE or CE).
    @param lmbd: weight of the sparsity-inducing penalty.
    @param gamma: weight of the time-smoothing penalty.
    @param batch_size: batch size.
    @param device: Torch device.
    @param alpha: alpha-parameter for the elastic-net (default: 0.5).
    @param verbose: print-outs enabled?
    @param train: training mode?
    @return: if train == False, returns generalised coefficient matrices and average losses incurred; otherwise, None.
    """
    if not train:
        coeffs_final = torch.zeros((predictors.shape[0], predictors.shape[1], predictors.shape[2],
                                    predictors.shape[2])).to(device)

    # Shuffle the data
    inds = np.arange(0, predictors.shape[0])
    if train:
        np.random.shuffle(inds)

    # Split into batches
    batch_split = np.arange(0, len(inds), batch_size)
    if len(inds) - batch_split[-1] < batch_size / 2:
        batch_split = batch_split[:-1]

    incurred_loss = 0
    incurred_base_loss = 0
    incurred_penalty = 0
    incurred_smoothness_penalty = 0
    for i in range(len(batch_split)):
        if i < len(batch_split) - 1:
            predictors_b = predictors[inds[batch_split[i]:batch_split[i + 1]], :, :]
            responses_b = responses[inds[batch_split[i]:batch_split[i + 1]], :]
            time_idx_b = time_idx[inds[batch_split[i]:batch_split[i + 1]]]
        else:
            predictors_b = predictors[inds[batch_split[i]:], :, :]
            responses_b = responses[inds[batch_split[i]:], :]
            time_idx_b = time_idx[inds[batch_split[i]:]]

        inputs = Variable(torch.tensor(predictors_b, dtype=torch.float)).float().to(device)
        targets = Variable(torch.tensor(responses_b, dtype=torch.float)).float().to(device)

        # Get the forecasts and generalised coefficients
        preds, coeffs = model(inputs=inputs)
        if not train:
            if i < len(batch_split) - 1:
                coeffs_final[inds[batch_split[i]:batch_split[i + 1]], :, :, :] = coeffs
            else:
                coeffs_final[inds[batch_split[i]:], :, :, :] = coeffs

        # Loss
        # Base loss
        base_loss = criterion(preds, targets)

        # Sparsity-inducing penalty term
        # coeffs.shape:     [T x K x p x p]
        penalty = (1 - alpha) * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=2), dim=0)) + \
                  alpha * torch.mean(torch.mean(torch.norm(coeffs, dim=1, p=1), dim=0))

        # Smoothing penalty term
        next_time_points = time_idx_b + 1
        inputs_next = Variable(torch.tensor(predictors[np.where(np.isin(time_idx, next_time_points))[0], :, :],
                                            dtype=torch.float)).float().to(device)
        preds_next, coeffs_next = model(inputs=inputs_next)
        penalty_smooth = torch.norm(coeffs_next - coeffs[np.isin(next_time_points, time_idx), :, :, :], p=2)

        loss = base_loss + lmbd * penalty + gamma * penalty_smooth

        # Incur loss
        incurred_loss += loss.data.cpu().numpy()
        incurred_base_loss += base_loss.data.cpu().numpy()
        incurred_penalty += lmbd * penalty.data.cpu().numpy()
        incurred_smoothness_penalty += gamma * penalty_smooth.data.cpu().numpy()

        if train:
            # Make an optimisation step
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    if verbose:
        print("Epoch " + str(epoch_num) + " : incurred loss " + str(incurred_loss) + "; incurred sparsity penalty " +
              str(incurred_penalty) + "; incurred smoothness penalty " + str(incurred_smoothness_penalty))

    if not train:
        return coeffs_final, incurred_loss / len(batch_split), incurred_base_loss / len(batch_split), \
               incurred_penalty / len(batch_split), incurred_smoothness_penalty / len(batch_split)
