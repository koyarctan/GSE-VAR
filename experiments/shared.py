"""Shared numerical metrics, transparent persistence, and model fitting.

Array order is always time, lag (oldest first), target, source. No pickle.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import f as f_distribution
import torch

from gse_var import GSEVAR, GSEVARTrainingConfig, fit_gse_var
from gse_var.data import construct_lagged_dataset

METHODS = ('GSE-VAR',)
SCHEMA_VERSION = 1


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    if any(np.asarray(v).dtype.hasobject for v in arrays.values()):
        raise TypeError('Object arrays / pickle are not permitted.')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(jsonable(v), ensure_ascii=False)
                             if isinstance(v, (dict, list, tuple)) else jsonable(v)
                             for k, v in row.items()})


def show_rows(rows, fields=None):
    rows = list(rows)
    if not rows:
        print('(no rows)')
        return
    fields = fields or list(rows[0])
    print('\t'.join(fields))
    for row in rows:
        print('\t'.join(f'{row[k]:.6g}' if isinstance(row.get(k), float)
                        else str(row.get(k, '')) for k in fields))


def environment(device):
    target = torch.device(device)
    if target.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use a CUDA-enabled PyTorch environment or DEVICE="cpu".')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return dict(python=platform.python_version(), numpy=np.__version__,
                torch=torch.__version__, scipy=scipy.__version__, device=str(device),
                matplotlib=importlib.metadata.version('matplotlib'),
                cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(target) if target.type == 'cuda' else None)


class Run:
    def __init__(self, project, experiment, config, data_manifest):
        self.project = Path(project).resolve()
        self.config = config
        self.device = config['device']
        sources = {}
        source_root = Path(__file__).resolve().parents[1]
        for root in (source_root / 'gse_var', source_root / 'experiments'):
            for path in sorted(root.rglob('*.py')):
                sources[path.relative_to(source_root).as_posix()] = path.read_text(encoding='utf-8')
        identity_config = {k:v for k,v in config.items() if k not in ('run_experiments', 'retry_errors', 'output_root')}
        self.spec = dict(schema_version=SCHEMA_VERSION, experiment=experiment,
                         config=identity_config, data=data_manifest, source_sha=digest(sources),
                         environment=environment(self.device))
        self.signature = digest(self.spec)[:16]
        output_root = Path(config.get('output_root', self.project/'results'))
        self.path = output_root / experiment / f"{config['mode']}_{self.signature}"
        self.path.mkdir(parents=True, exist_ok=True)
        self.spec = jsonable(self.spec)
        manifest = self.path / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != self.spec:
            raise ValueError('Manifest mismatch; refusing to reuse results.')
        write_json(manifest, self.spec)
        write_json(self.path / 'source_snapshot.json', sources)
        write_json(self.path / 'array_schema.json', dict(
            lag_order='oldest_first; lag index K-1 is lag 1', matrix_axes=['target', 'source'],
            coefficients=['time', 'lag', 'target', 'source'],
            jacobian=['time', 'lag', 'target', 'source'], gate=['lag', 'target', 'source'],
            selection='fixed Bernoulli mask at probability 0.5',
            files=dict(fits='final arrays, full histories and weights',
                       checkpoints='weights every CHECK_EVERY epochs; forward reconstruction, not optimizer resume',
                       datasets='original/processed series, truth, train/evaluation arrays',
                       tables='UTF-8 CSV long-form results', figures='PDF and PNG, regenerable')))
        print('Result directory:', self.path, flush=True)

    def log(self, text):
        line = f'[{datetime.now(timezone.utc).isoformat()}] {text}'
        with (self.path / 'progress.log').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
        print(line, flush=True)

    @contextmanager
    def lock(self):
        lock = self.path / 'RUNNING.lock'
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f'Run already locked: {lock}. Check the process before removing a stale lock.') from None
        with os.fdopen(fd, 'w') as stream:
            stream.write(f'{platform.node()} pid={os.getpid()}\n')
        try:
            yield
        finally:
            lock.unlink(missing_ok=True)

    def cached(self, key, compute):
        ident = digest(key)
        path = self.path / 'fits' / f'{ident}.npz'
        if path.exists():
            data = read_npz(path)
            if json.loads(str(data['fit_spec'])) != jsonable(key):
                raise ValueError('Fit specification mismatch.')
            return data
        if not self.config.get('run_experiments', True):
            raise FileNotFoundError(path)
        error_path = self.path / 'errors' / f'{ident}.json'
        if error_path.exists() and not self.config.get('retry_errors', False):
            raise RuntimeError(f'Previously failed fit; inspect {error_path} before enabling retry_errors.')
        self.log('START ' + json.dumps(jsonable(key), ensure_ascii=False))
        started = time.perf_counter()
        try:
            result = compute(ident)
            result['fit_spec'] = np.array(json.dumps(jsonable(key), sort_keys=True))
            result['runtime_seconds'] = np.array(time.perf_counter() - started)
            write_npz(path, **result)
            self.log(f'DONE {ident} ({float(result["runtime_seconds"]):.1f}s)')
            return result
        except Exception as error:
            write_json(error_path, dict(fit_spec=key, error=repr(error), traceback=traceback.format_exc()))
            self.log(f'ERROR {ident}: {error!r}')
            raise


def lagged(z, order):
    data = construct_lagged_dataset(np.asarray(z, np.float32), order)
    return data.predictors, data.responses


def state_arrays(model):
    return {'state__' + k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def restore_model(method, profile, p, arrays, device='cpu'):
    if method not in ('GSE-VAR', 'GSE-DisARM'):
        raise ValueError(method)
    model = GSEVAR(num_vars=p, order=profile['order'], hidden_layer_size=profile['hidden'],
                   num_hidden_layers=profile.get('layers', 1),
                   gate_probability_init=profile.get('probability_init', .5)).to(device)
    model.load_state_dict({k.removeprefix('state__'): torch.as_tensor(v, device=device)
                           for k, v in arrays.items() if k.startswith('state__')}, strict=True)
    model.eval()
    return model


def snapshot(model, x, *, jacobian=False):
    device = next(model.parameters()).device
    prediction, coefficients, jacobians = [], [], []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, len(x), 128):
            inputs = torch.as_tensor(x[start:start+128], dtype=torch.float32, device=device)
            with torch.no_grad():
                predicted, coefficient = model(inputs)
            prediction.append(predicted.detach().cpu().numpy())
            coefficients.append(coefficient.detach().cpu().numpy())
            if jacobian:
                inputs = inputs.detach().requires_grad_(True)
                with torch.enable_grad():
                    y, _ = model(inputs)
                    parts = [torch.autograd.grad(y[:, i].sum(), inputs, retain_graph=i+1<y.shape[1])[0]
                             for i in range(y.shape[1])]
                jacobians.append(torch.stack(parts, dim=2).detach().cpu().numpy())
        out = dict(predictions=np.concatenate(prediction), coefficients=np.concatenate(coefficients))
        if jacobian:
            out['jacobian'] = np.concatenate(jacobians)
        for key in ('causal_gate', 'gate_probabilities'):
            value = getattr(model, key, None)
            if value is not None:
                out['gate' if key == 'causal_gate' else key] = value.detach().cpu().numpy().copy()
        if not all(np.isfinite(v).all() for v in out.values()):
            raise FloatingPointError('Nonfinite inference arrays.')
        return out
    finally:
        model.train(was_training)


def gse_profile(method, order, p, hidden, budget, epochs, **overrides):
    if method not in ('GSE-VAR', 'GSE-DisARM'):
        raise ValueError(method)
    profile = dict(order=order, hidden=hidden, layers=1, epochs=epochs, lr=.003,
                   gate_lr=.03, smooth=.01, decay=.001, budget=float(budget),
                   l0=float(budget/(p*p*order)), probability_init=.5)
    profile.update(overrides)
    return profile


def fit_gse(run, case, method, profile, seed, score=None):
    key = dict(dataset=case['id'], method=method, profile=profile, seed=int(seed))
    x_eval, y_eval = case['x_eval'], case['y_eval']
    checkpoint_x, checkpoint_y = (lagged(case['train'], profile['order']) if score is False
                                  else (x_eval, y_eval))
    def compute(ident):
        history, checkpoint_rows, checkpoint_gates, checkpoint_probabilities = [], [], [], []
        every = run.config['check_every']
        def observe(epoch, model, record):
            history.append(dict(epoch=epoch, **record))
            if epoch % every and epoch != profile['epochs']:
                return
            arrays = snapshot(model, checkpoint_x)
            metrics = score(arrays) if callable(score) else dict(mse=float(np.mean((arrays['predictions']-checkpoint_y)**2)))
            checkpoint_rows.append(dict(epoch=epoch, **metrics))
            checkpoint_gates.append(arrays['gate'])
            if 'gate_probabilities' in arrays:
                checkpoint_probabilities.append(arrays['gate_probabilities'])
            write_npz(run.path / 'checkpoints' / ident / f'{epoch:06d}.npz',
                      epoch=np.array(epoch), **state_arrays(model))
            run.log(f'checkpoint {ident} epoch={epoch} {metrics}')
        shared = dict(order=profile['order'], hidden_layer_size=profile['hidden'],
                      num_hidden_layers=profile['layers'], max_epochs=profile['epochs'],
                      batch_size=len(case['train'])-profile['order'], learning_rate=profile['lr'],
                      gate_learning_rate=profile['gate_lr'], lambda_smooth=profile['smooth'],
                      coefficient_weight_decay=profile['decay'], lambda_jacobian=0.,
                      seed=int(seed), shuffle=False, verbose=0, device=run.device)
        if method not in ('GSE-VAR', 'GSE-DisARM'):
            raise ValueError(method)
        config = GSEVARTrainingConfig(**shared, lambda_l0=profile['l0'],
                                     gate_probability_init=profile['probability_init'])
        fitted = fit_gse_var(case['train'], config, epoch_callback=observe)
        result = snapshot(fitted.model, x_eval, jacobian=True)
        result.update(state_arrays(fitted.model))
        history_fields = list(history[0])
        cp_fields = list(checkpoint_rows[0])
        result.update(history=np.array([[r[k] for k in history_fields] for r in history]),
                      history_fields=np.array(history_fields),
                      checkpoint_metrics=np.array([[r[k] for k in cp_fields] for r in checkpoint_rows]),
                      checkpoint_fields=np.array(cp_fields), checkpoint_gates=np.stack(checkpoint_gates),
                      parameters=np.array(sum(p.numel() for p in fitted.model.parameters())))
        if checkpoint_probabilities:
            result['checkpoint_probabilities'] = np.stack(checkpoint_probabilities)
        write_csv(run.path / 'tables' / f'{ident}_history.csv', history)
        write_csv(run.path / 'tables' / f'{ident}_checkpoints.csv', checkpoint_rows)
        return result
    return run.cached(key, compute)


def bh_select(pvalues, q=.05):
    values = np.asarray(pvalues).ravel()
    order = np.argsort(values)
    accept = values[order] <= q * np.arange(1, len(values)+1)/len(values)
    selected = np.zeros(len(values), bool)
    if accept.any():
        selected[order[:np.flatnonzero(accept)[-1]+1]] = True
    return selected.reshape(np.shape(pvalues))


def graph_metrics(truth, graph, strength=None):
    mask = ~np.eye(len(truth), dtype=bool)
    a, b = np.asarray(truth, bool)[mask], np.asarray(graph, bool)[mask]
    tp, fp, fn, tn = (int(v.sum()) for v in (a & b, ~a & b, a & ~b, ~a & ~b))
    result = dict(tp=tp, fp=fp, fn=fn, tn=tn, fpr=fp/max(fp+tn,1), edges=int(b.sum()), density=float(b.mean()),
                  precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
                  f1=2*tp/max(2*tp+fp+fn, 1), shd=fp+fn,
                  balanced_accuracy=.5*(tp/max(tp+fn, 1)+tn/max(tn+fp, 1)))
    if strength is not None:
        result.update(score_metrics(a, np.asarray(strength)[mask]))
    return result


def score_metrics(truth, score):
    y, s = np.asarray(truth, bool).ravel(), np.asarray(score, float).ravel()
    if not np.isfinite(s).all() or not y.any() or y.all():
        raise ValueError('AP/AUROC require finite scores and both classes.')
    inds = np.argsort(-s, kind='stable')
    ys, ss = y[inds], s[inds]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss)-1]
    tp = np.r_[0, np.cumsum(ys)[ends]].astype(float)
    fp = np.r_[0, (ends+1)-np.cumsum(ys)[ends]].astype(float)
    recall, fpr = tp/y.sum(), fp/(~y).sum()
    precision = np.divide(tp, tp+fp, out=np.ones_like(tp), where=tp+fp>0)
    ap = float(np.sum(np.diff(recall)*precision[1:]))
    auroc = float(np.trapezoid(recall, fpr) if hasattr(np, 'trapezoid') else np.trapz(recall, fpr))
    return dict(auprc=float(np.sum(np.diff(recall)*(precision[:-1]+precision[1:])/2)), average_precision=ap, auroc=auroc)


def support(arrays):
    return np.asarray(arrays['gate']) != 0


def strength(arrays):
    return np.median(np.abs(arrays['coefficients']), axis=0).max(axis=0)


def sign_metrics(truth, arrays, graph):
    signed = np.sign(np.median(arrays['coefficients'], axis=0).sum(axis=0))*graph
    mask = ~np.eye(len(truth), dtype=bool)
    y, pred = np.asarray(truth)[mask], signed[mask]
    out = {}
    for label, val in [('pos', 1), ('neg', -1)]:
        a, b = y == val, pred == val
        if not a.any() or a.all():
            out['ba_' + label] = float('nan')
        else:
            out['ba_' + label] = .5*(np.mean(b[a])+np.mean(~b[~a]))
    out['mean_sign_ba'] = (out['ba_pos']+out['ba_neg'])/2
    real = mask & (truth != 0)
    out['true_edge_sign_accuracy'] = float(np.mean(signed[real] == truth[real]))
    return out, signed.astype(np.int8)


def linear_fit(train, x_eval, order, group=0., l1=0., max_iterations=10000, tolerance=1e-7):
    x, y = lagged(train, order)
    n, p = y.shape
    flat, future = x.reshape(n, -1).astype(float), x_eval.reshape(len(x_eval), -1).astype(float)
    mx, my = flat.mean(0), y.mean(0)
    xc, yc = flat-mx, y-my
    if group == 0 and l1 == 0:
        w = np.linalg.lstsq(xc, yc, rcond=None)[0]
        residual = yc-xc@w
        rank = np.linalg.matrix_rank(np.column_stack([np.ones(n), flat]))
        df = n-rank
        if df <= 0:
            raise ValueError('Insufficient residual degrees of freedom for conditional F tests.')
        pvalues = np.ones((p, p))
        for source in range(p):
            keep = np.ones(order*p, bool)
            keep[np.arange(order)*p+source] = False
            reduced = xc[:, keep]
            difference_rank = rank - np.linalg.matrix_rank(np.column_stack([np.ones(n), reduced]))
            if difference_rank <= 0:
                continue
            rres = yc-reduced@np.linalg.lstsq(reduced, yc, rcond=None)[0]
            f = np.maximum((np.square(rres).sum(0)-np.square(residual).sum(0))/difference_rank, 0)
            f /= np.maximum(np.square(residual).sum(0)/df, np.finfo(float).tiny)
            pvalues[:, source] = f_distribution.sf(f, difference_rank, df)
        graph = bh_select(pvalues)
        iterations, fixed_point_residual = 1, 0.
    else:
        lipschitz = 2*np.linalg.norm(xc, 2)**2/(n*p)
        step = 1/max(lipschitz, 1e-12)
        def prox(v):
            v = np.sign(v)*np.maximum(np.abs(v)-step*l1, 0.)
            shaped = v.reshape(order, p, p)
            norm = np.linalg.norm(shaped, axis=0, keepdims=True)
            return (shaped*np.maximum(1-step*group/np.maximum(norm, 1e-300), 0.)).reshape(v.shape)
        w, extrapolated, t = np.zeros((order*p, p)), np.zeros((order*p, p)), 1.
        converged = False
        for iterations in range(1, max_iterations+1):
            previous = w
            w = prox(extrapolated-step*2*xc.T@(xc@extrapolated-yc)/(n*p))
            nt = .5*(1+np.sqrt(1+4*t*t))
            extrapolated = w+(t-1)/nt*(w-previous)
            t = nt
            if iterations % 10 == 0:
                fixed = prox(w-step*2*xc.T@(xc@w-yc)/(n*p))
                fixed_point_residual = np.linalg.norm(w-fixed)/step/max(1., np.linalg.norm(w))
                if fixed_point_residual < tolerance:
                    converged = True
                    break
        if not converged:
            raise RuntimeError(f'Linear SGL not converged: {fixed_point_residual:g}')
        graph = (w.reshape(order, p, p) != 0).any(0).T
        pvalues = np.ones((p, p))
    coefficient = w.reshape(order, p, p).transpose(0, 2, 1)
    intercept = my-mx@w
    return dict(predictions=(future@w+intercept).astype(float),
                coefficients=np.broadcast_to(coefficient, (len(future), *coefficient.shape)).copy(),
                jacobian=np.broadcast_to(coefficient, (len(future), *coefficient.shape)).copy(),
                gate=(coefficient != 0).astype(np.int8), graph=graph.astype(np.int8),
                pvalues=pvalues, intercept=intercept, beta=w,
                iterations=np.array(iterations), fixed_point_residual=np.array(fixed_point_residual))
