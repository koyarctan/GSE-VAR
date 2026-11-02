"""Independent tuning and paired evaluation for Lorenz96 and linear VAR."""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch

from gse_var import CMLP, CMLPTrainingConfig, fit_cmlp
from .shared import (Run, METHODS, digest, write_json, write_npz, read_npz, write_csv,
                     show_rows, lagged, state_arrays, snapshot, fit_gse, gse_profile,
                     graph_metrics, sign_metrics, strength, linear_fit)
from .reference.ngc.synthetic import simulate_lorenz_96
from .reference.gvar.linear_examples import generate_linear_example_1
from .reference.gvar.senn import SENNGC
from .reference.gvar.epoch import run_epoch as gvar_epoch


def default_config(kind, mode='full', device='cuda:0'):
    if kind not in ('lorenz96', 'linear_var') or mode not in ('full', 'smoke'):
        raise ValueError((kind, mode))
    small = mode == 'smoke'
    return dict(mode=mode, device=device, run_experiments=True, retry_errors=False,
                lengths=[48, 64] if small else [250, 500],
                order=(2 if small else 5) if kind == 'lorenz96' else 1,
                p=(6 if small else 20) if kind == 'lorenz96' else 4,
                forcing=40. if kind == 'lorenz96' else None,
                burn_in=(20 if small else 1000) if kind == 'lorenz96' else 0,
                holdout=(12 if small else 200) if kind == 'lorenz96' else 0,
                delta_t=.05 if kind == 'lorenz96' else 1., noise_sd=.1 if kind == 'lorenz96' else .4,
                epochs=3 if small else 10000, check_every=2 if small else 100,
                hidden=[4] if small else [8, 32],
                budgets=[0., .1] if small else [0., .01, .03, .1, .3, 1., 3., 10., 30.],
                pilot_seeds=[10000] if small else [10000, 10001, 10002], pilot_training_seed=142,
                main_seeds=([0, 1] if kind == 'lorenz96' else [42, 43]) if small else (list(range(5)) if kind == 'lorenz96' else list(range(42, 52))),
                ngc_epochs=3 if small else 5000, ngc_hidden=4 if small else 100,
                ngc_lambdas=[0., .3] if small else [0., .01, .05, .1, .2, .3, .4, .5, 1.],
                gvar_epochs=3 if small else 1000, gvar_hidden=4 if small else 50,
                gvar_batch=16 if small else 64,
                gvar_lambdas=[.2] if kind == 'linear_var' else ([0., 3.] if small else [0., 1., 3.]),
                gvar_gammas=[.5] if kind == 'linear_var' else ([0.] if small else [0., .01, .025]))


class SyntheticExperiment:
    def __init__(self, project, kind, config):
        self.kind, self.config = kind, config
        if kind == 'linear_var' and (config['p'] != 4 or config['order'] != 1 or config['noise_sd'] != .4):
            raise ValueError('The fixed author linear generator requires p=4, order=1, noise_sd=.4.')
        if set(config['pilot_seeds']) & set(config['main_seeds']):
            raise ValueError('Pilot/evaluation data seeds overlap.')
        self.methods = [*METHODS, *(['NGC'] if kind == 'lorenz96' else []), 'GVAR', 'Linear-VAR']
        self.run = Run(project, '12_Lorenz96' if kind == 'lorenz96' else '13_LinearVAR', config,
                       dict(generator=kind, ngc_commit='c3263d40433aaf3acc94c27a0c1abf9b8e9fcedf',
                            gvar_commit='e06268f87fbe923580af14b2cab26399191747a6'))

    def case(self, length, seed):
        c, p, order = self.config, self.config['p'], self.config['order']
        generation = {k:c[k] for k in ('lengths','holdout','p','order','forcing','delta_t','noise_sd','burn_in')}
        key = digest(dict(kind=self.kind, length=length, seed=seed, generation=generation))
        path = self.run.path / 'datasets' / f'{key}.npz'
        if path.exists():
            arrays = read_npz(path)
        else:
            count = max(c['lengths']) + (c['holdout'] if self.kind == 'lorenz96' else 0)
            if self.kind == 'lorenz96':
                raw, truth = simulate_lorenz_96(p, count, F=c['forcing'], delta_t=c['delta_t'],
                                   sd=c['noise_sd'], burn_in=c['burn_in'], seed=seed)
                a = np.zeros((p, p))
                actual_dt = (count+c['burn_in'])*c['delta_t']/(count+c['burn_in']-1)
            else:
                # Recover coefficients from the same RandomState draws as the author's generator.
                rng, a = np.random.RandomState(seed), np.zeros((p, p))
                values = []
                for _ in range(8):
                    interval = (-.8, -.2) if rng.uniform(0, 1, 1).item() <= .5 else (.2, .8)
                    values.append(rng.uniform(*interval, size=1).item())
                a[0, 0], a[1, 1], a[1, 0], a[2, 2] = values[:4]
                a[2, 1], a[3, 3], a[3, 1], a[3, 2] = values[4:]
                series, truth, signs = generate_linear_example_1(1, count, seed=seed)
                raw, actual_dt = series[0], 1.
                if not np.array_equal(np.sign(a), signs[0]):
                    raise AssertionError('Coefficient reconstruction differs from author generator.')
            mean, scale = raw[:length].mean(0), raw[:length].std(0)
            if (scale <= 0).any() or not np.isfinite(raw).all():
                raise FloatingPointError('Invalid generated series.')
            z = ((raw-mean)/scale).astype(np.float32)
            x, y = lagged(z[:length], order)
            all_x, all_y = lagged(z, order)
            arrays = dict(raw=raw, standardized=z, train=z[:length], mean=mean, scale=scale,
                          truth=truth.astype(np.int8), truth_signed=np.sign(a).astype(np.int8),
                          true_coefficient=a, true_standardized=a*scale[None, :]/scale[:, None],
                          x_eval=x, y_eval=y, hold_x=all_x[length-order:], hold_y=all_y[length-order:],
                          actual_dt=np.array(actual_dt), names=np.array([f'X{i+1}' for i in range(p)]
                                               if self.kind == 'lorenz96' else ['x', 'w', 'y', 'z']))
            write_npz(path, **arrays)
        return dict(arrays, id=key, length=length, generation_seed=seed)

    def candidates(self, method, length=None):
        c = self.config
        grid = c.get('regularization_grid', {}).get(str(length), {})
        if method in METHODS:
            return [gse_profile(method, c['order'], c['p'], hidden, budget, c['epochs'])
                    for hidden, budget in itertools.product(c['hidden'], grid.get(method, c['budgets']))]
        if method == 'NGC':
            return [dict(order=c['order'], hidden=c['ngc_hidden'], layers=1,
                         epochs=c['ngc_epochs'], lam=lam, lr=.01, ridge=.01) for lam in grid.get(method, c['ngc_lambdas'])]
        if method == 'GVAR':
            return [dict(order=c['order'], hidden=c['gvar_hidden'], layers=1 if c['mode']=='smoke' else 2,
                         epochs=c['gvar_epochs'], lam=lam, smooth=gamma, lr=.0001, batch=c['gvar_batch'])
                    for lam, gamma in itertools.product(c['gvar_lambdas'], c['gvar_gammas'])]
        return [dict(order=c['order'], q=.05)]

    def fit_gvar(self, case, profile, seed, reverse=False):
        key = dict(dataset=case['id'], method='GVAR', profile=profile, seed=seed, reverse=reverse)
        def compute(ident):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            series = case['train'][::-1].copy() if reverse else case['train']
            x, y = lagged(series, profile['order'])
            model = SENNGC(self.config['p'], profile['order'], profile['hidden'], profile['layers'],
                           torch.device(self.run.device)).to(self.run.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=profile['lr'])
            history = []
            for epoch in range(1, profile['epochs']+1):
                gvar_epoch(epoch, model, optimizer, x, y, np.arange(len(x)), torch.nn.MSELoss(),
                           profile['lam'], profile['smooth'], profile['batch'],
                           torch.device(self.run.device), verbose=False)
                if not all(torch.isfinite(v).all() for v in model.parameters()):
                    raise FloatingPointError(f'GVAR nonfinite: epoch={epoch}')
                if epoch % self.config['check_every'] == 0 or epoch == profile['epochs']:
                    with torch.no_grad():
                        _, loss, mse, penalty, smooth = gvar_epoch(epoch, model, optimizer, x, y,
                                    np.arange(len(x)), torch.nn.MSELoss(), profile['lam'], profile['smooth'],
                                    profile['batch'], torch.device(self.run.device), verbose=False, train=False)
                    history.append(dict(epoch=epoch, loss=float(loss), mse=float(mse),
                                        penalty=float(penalty), smooth=float(smooth)))
                    write_npz(self.run.path/'checkpoints'/ident/f'{epoch:06d}.npz', **state_arrays(model))
                    self.run.log(f'checkpoint {ident} epoch={epoch} GVAR loss={loss:g}')
            arrays = snapshot(model, x, jacobian=True)
            arrays.update(state_arrays(model))
            arrays.update(history=np.array([list(r.values()) for r in history]), history_fields=np.array(list(history[0])),
                          parameters=np.array(sum(p.numel() for p in model.parameters())))
            write_csv(self.run.path/'tables'/f'{ident}_history.csv', history)
            return arrays
        return self.run.cached(key, compute)

    def fit_ngc(self, case, profile, seed):
        key = dict(dataset=case['id'], method='NGC', profile=profile, seed=seed)
        def compute(ident):
            torch.manual_seed(seed)
            model = CMLP(self.config['p'], profile['order'], profile['hidden'], profile['layers']).to(self.run.device)
            histories = []
            every = self.config['check_every']
            for start in range(0, profile['epochs'], every):
                # ISTA is stateless; public full-batch fits in contiguous blocks preserve its trajectory.
                config = CMLPTrainingConfig(order=profile['order'], hidden_layer_size=profile['hidden'],
                        num_hidden_layers=1, max_epochs=min(every, profile['epochs']-start),
                        batch_size=len(case['x_eval']), learning_rate=profile['lr'], lambda_ngc=profile['lam'],
                        ridge_lambda=profile['ridge'], regularizer='hierarchical_group_lasso',
                        optimizer='ista', seed=None, shuffle=False, causal_threshold=0., verbose=0, device=self.run.device)
                fitted = fit_cmlp(case['train'], config, model=model)
                for i in range(config.max_epochs):
                    row = dict(epoch=start+i+1, **{k: v[i] for k, v in fitted.history.items()})
                    if not all(np.isfinite(v) for v in row.values()):
                        raise FloatingPointError('NGC nonfinite history.')
                    histories.append(row)
                write_npz(self.run.path/'checkpoints'/ident/f'{start+config.max_epochs:06d}.npz', **state_arrays(model))
                self.run.log(f'checkpoint {ident} epoch={start+config.max_epochs} NGC')
            with torch.no_grad():
                predictions = model(torch.as_tensor(case['x_eval'], device=self.run.device)).cpu().numpy()
                score = model.gc_strength(ignore_lag=True).cpu().numpy()
            arrays = dict(predictions=predictions, graph=(score != 0).astype(np.int8), strength=score,
                          history=np.array([list(r.values()) for r in histories]), history_fields=np.array(list(histories[0])),
                          parameters=np.array(sum(p.numel() for p in model.parameters())), **state_arrays(model))
            write_csv(self.run.path/'tables'/f'{ident}_history.csv', histories)
            return arrays
        return self.run.cached(key, compute)

    def evaluate(self, case, method, profile, seed):
        if method in METHODS:
            arrays = fit_gse(self.run, case, method, profile, seed)
            graph, score = (arrays['gate'] != 0).any(0), strength(arrays)
        elif method == 'NGC':
            arrays = self.fit_ngc(case, profile, seed)
            graph, score = arrays['graph'], arrays['strength']
        elif method == 'GVAR':
            arrays = self.fit_gvar(case, profile, seed)
            reverse = self.fit_gvar(case, profile, seed, reverse=True)
            score = strength(arrays)
            graph, degenerate, agreement = trgc(score, strength(reverse))
            arrays = dict(arrays, trgc_agreement=np.array(agreement), trgc_degenerate=np.array(degenerate),
                          reverse_strength=strength(reverse), reverse_fit_spec=reverse['fit_spec'],
                          runtime_seconds=arrays['runtime_seconds']+reverse['runtime_seconds'])
        else:
            key = dict(dataset=case['id'], method=method, profile=profile, seed=0)
            arrays = self.run.cached(key, lambda _: linear_fit(case['train'], case['x_eval'], self.config['order']))
            graph, score = arrays['graph'], -arrays['pvalues']
        metrics = graph_metrics(case['truth'], graph, score)
        if self.kind == 'linear_var':
            signed_metrics, signed = sign_metrics(case['truth_signed'], arrays, graph)
            metrics.update(signed_metrics)
        else:
            signed = np.zeros_like(graph, dtype=np.int8)
        metrics['training_mse'] = float(np.mean((arrays['predictions']-case['y_eval'])**2))
        metrics['runtime_seconds'] = float(arrays['runtime_seconds'])
        return metrics, dict(arrays, graph=np.asarray(graph, np.int8), signed=signed, strength=score)

    def tune(self):
        path = self.run.path/'selection.json'
        if path.exists():
            return json.loads(path.read_text(encoding='utf-8'))
        choices, rows = {}, []
        criterion = 'f1' if self.kind == 'lorenz96' else 'mean_sign_ba'
        for length in self.config['lengths']:
            choices[str(length)] = {}
            for method in self.methods:
                candidates = self.candidates(method, length)
                if method == 'Linear-VAR' or self.kind == 'linear_var' and method == 'GVAR':
                    choices[str(length)][method] = candidates[0]
                    continue
                valid = []
                for profile in candidates:
                    samples = []
                    for seed in self.config['pilot_seeds']:
                        try:
                            metric, _ = self.evaluate(self.case(length, seed), method, profile,
                                                      self.config['pilot_training_seed'])
                            rows.append(dict(T=length, method=method, profile=profile, seed=seed, status='ok', **metric))
                            samples.append(metric)
                        except (FloatingPointError, RuntimeError, ValueError) as error:
                            rows.append(dict(T=length, method=method, profile=profile, seed=seed, status='error', error=repr(error)))
                        write_csv(self.run.path/'pilot_metrics.csv', rows)
                    if len(samples) == len(self.config['pilot_seeds']) and all(np.isfinite(m[criterion]) for m in samples):
                        valid.append((float(np.mean([m[criterion] for m in samples])),
                                      float(np.mean([m['edges'] for m in samples])), profile))
                if not valid:
                    raise RuntimeError(f'No complete finite pilot configuration: T={length}, {method}')
                best = sorted(valid, key=lambda row: (-row[0], row[1], row[2].get('hidden', 0), digest(row[2])))[0]
                choices[str(length)][method] = best[2]
                self.run.log(f'FREEZE T={length} {method} {criterion}={best[0]:.6f} {best[2]}')
        write_json(path, choices)
        return choices

    def evaluate_main(self, choices):
        index, records = [], []
        for length in self.config['lengths']:
            for replicate, seed in enumerate(self.config['main_seeds']):
                case = self.case(length, seed)
                training_seed = 42+replicate if self.kind == 'lorenz96' else 42
                for method in self.methods:
                    profile = choices[str(length)][method]
                    base = dict(T=length, replicate=replicate, generation_seed=seed, training_seed=training_seed, method=method)
                    try:
                        metric, arrays = self.evaluate(case, method, profile, training_seed)
                        key = digest(base)
                        file = self.run.path/'evaluations'/f'{key}.npz'
                        write_npz(file, **arrays)
                        records.append(dict(**base, status='ok', **metric))
                        index.append(dict(**base, dataset=f'datasets/{case["id"]}.npz', result=f'evaluations/{key}.npz', profile=profile))
                    except (FloatingPointError, RuntimeError, ValueError) as error:
                        records.append(dict(**base, status='error', error=repr(error)))
                    write_csv(self.run.path/'main_metrics.csv', records)
                    write_json(self.run.path/'main_index.json', index)
        failed = sum(row['status'] != 'ok' for row in records)
        write_json(self.run.path/'completion.json', dict(expected=len(records), complete=len(index), failed=failed))
        if failed:
            raise RuntimeError(f'{failed} failed evaluation fits. Inspect main_metrics.csv; no incomplete aggregate is published.')
        show_rows(records, ['T', 'method', 'replicate', 'auprc', 'f1'] + (['mean_sign_ba'] if self.kind == 'linear_var' else []))
        return records


def trgc(forward, reverse, quantiles=20):
    mask = ~np.eye(len(forward), dtype=bool)
    graphs, agreements = [], []
    for q in np.linspace(0, 1, quantiles):
        a, b = forward >= np.quantile(forward, q), (reverse >= np.quantile(reverse, q)).T
        av, bv = a[mask], b[mask]
        if av.all() or not av.any() or bv.all() or not bv.any():
            agreement = 0.
        else:
            def ba(x, y):
                return .5*(y[x].mean()+(~y[~x]).mean())
            agreement = .5*(ba(av, bv)+ba(bv, av))
        graphs.append(a)
        agreements.append(float(agreement))
    best = int(np.argmax(agreements))
    return graphs[best].astype(np.int8), max(agreements) <= 0, agreements
