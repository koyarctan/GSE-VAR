"""Validation-calibrated sparse models for category-level empirical analysis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import identity, prepare_data, make_cases, save_json

METHODS = ('GSE-VAR', 'Linear-SGL')
LABELS = {'GSE-VAR': 'GSE-VAR', 'Linear-SGL': 'Linear VAR(SGL)'}
CHECKPOINT_FIELDS = ('epoch', 'mse', 'se', 'edges', 'lag_routes', 'training_mse')


def study_root(project, config, manifest):
    settings = {k: v for k, v in config.items() if k != 'devices'}
    study = identity(dict(config=settings, data=manifest))[:16]
    return Path(project) / 'results' / '11_Favorita' / study


def validation_metrics(predictions, observed, gate, blocks):
    errors = np.mean(np.square(predictions-observed), axis=1)
    if len(errors) < blocks or blocks < 2 or not np.isfinite(errors).all():
        raise ValueError('Finite validation errors and at least two time blocks are required.')
    block_means = np.array([part.mean() for part in np.array_split(errors, blocks)])
    active = np.asarray(gate) != 0
    offdiag = ~np.eye(active.shape[-1], dtype=bool)
    return dict(mse=float(errors.mean()), se=float(block_means.std(ddof=1)/np.sqrt(blocks)),
                edges=int(active.any(0)[offdiag].sum()), lag_routes=int(active[:, offdiag].sum()))


def choose_candidate(rows):
    """Select minimum validation MSE; break ties by epoch and candidate."""
    if not rows or any(not np.isfinite(r['mse']) for r in rows):
        raise ValueError('Finite validation candidates are required.')
    chosen = min(rows, key=lambda r: (r['mse'], r['epoch'], r['candidate']))
    return dict(chosen, minimum_validation_mse=chosen['mse'], threshold=chosen['mse'])


def profile_for(config, method, budget, p, rates=None):
    k = config['order']
    if method == 'Linear-SGL':
        return dict(order=k, group=float(budget), l1=config['linear_l1'], budget=float(budget))
    if method != 'GSE-VAR':
        raise ValueError(method)
    rates = rates or config['learning_rates'][0]
    return dict(order=k, hidden=config['hidden'], layers=config['layers'], epochs=config['epochs'],
                lr=rates['lr'], gate_lr=rates['gate_lr'], smooth=config['smooth'], decay=config['decay'],
                budget=float(budget), l0=float(budget/(p*p*k)), probability_init=config['probability_init'])


def candidate_profiles(config, method, p):
    if method == 'Linear-SGL':
        return [profile_for(config, method, group, p) for group in config['linear_groups']]
    return [profile_for(config, method, budget, p, rates)
            for rates in config['learning_rates'] for budget in config['disarm_budgets']]


def should_check(epoch, last_epoch, config):
    return (epoch == 1 or epoch == last_epoch or epoch % config['check_every'] == 0 or
            (epoch <= config['early_check_until'] and epoch % config['early_check_every'] == 0))


def validate_config(config):
    for key in ('order', 'hidden', 'layers', 'epochs', 'check_every', 'early_check_every', 'early_check_until'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer.')
    if config['epochs'] > 5000:
        raise ValueError('max_epochs must not exceed 5000.')
    if config['validation_blocks'] < 2 or not 0.5 < config['sign_fraction'] <= 1:
        raise ValueError('Invalid validation blocks or sign fraction.')
    if not 0 < config['min_nonzero_fraction'] <= 1 or not 0 < config['probability_init'] < 1:
        raise ValueError('Invalid data-quality or initial gate probability.')
    if config['tuning_seed'] not in config['seeds'] or len(set(config['seeds'])) != len(config['seeds']):
        raise ValueError('seeds must contain the tuning seed exactly once.')
    for key in ('disarm_budgets', 'linear_groups'):
        values = config[key]
        if len(values) != 10 or len(set(values)) != 10 or not np.isfinite(values).all() or min(values) < 0:
            raise ValueError(f'{key} must contain 10 distinct nonnegative finite candidates.')
    rates = config['learning_rates']
    if not rates or len({(r['lr'], r['gate_lr']) for r in rates}) != len(rates):
        raise ValueError('Learning-rate pairs must be nonempty and distinct.')
    if any(not np.isfinite(r[k]) or r[k] <= 0 for r in rates for k in ('lr', 'gate_lr')):
        raise ValueError('Learning rates must be finite and positive.')
    if config['devices']['Linear-SGL'] != 'cpu':
        raise ValueError('Linear VAR(SGL) uses the CPU implementation.')


def linear_fit(segments, inputs, group, l1, tolerance=1e-7):
    from gse_var.data import construct_lagged_dataset
    k, p = inputs.shape[1:]
    data = construct_lagged_dataset(segments, k)
    x = data.predictors.reshape(len(data.responses), -1).astype(float)
    y = data.responses.astype(float)
    mx, my = x.mean(0), y.mean(0)
    x, y = x-mx, y-my
    n = len(y)
    gram, cross = x.T@x/n, x.T@y/n
    step = p/max(2*np.linalg.eigvalsh(gram)[-1], 1e-12)
    def prox(a):
        a = np.sign(a)*np.maximum(np.abs(a)-step*l1, 0)
        a = a.reshape(k, p, p)
        norm = np.linalg.norm(a, axis=0, keepdims=True)
        return (a*np.maximum(1-step*group/np.maximum(norm, 1e-300), 0)).reshape(k*p, p)
    w, v, momentum = np.zeros((k*p, p)), np.zeros((k*p, p)), 1.
    for iteration in range(1, 30001):
        previous = w
        w = prox(v-step*2*(gram@v-cross)/p)
        next_momentum = (1+np.sqrt(1+4*momentum**2))/2
        v = w+(momentum-1)/next_momentum*(w-previous)
        momentum = next_momentum
        if iteration % 10 == 0:
            fixed = prox(w-step*2*(gram@w-cross)/p)
            residual = float(np.linalg.norm(w-fixed)/step/max(1., np.linalg.norm(w)))
            if residual < tolerance:
                break
    else:
        raise RuntimeError(f'Linear SGL did not converge: {residual:g}')
    coefficient = w.reshape(k, p, p).transpose(0, 2, 1)
    return dict(beta=w, intercept=my-mx@w, gate=(coefficient != 0).astype(np.int8),
                iterations=np.array(iteration), fixed_point_residual=np.array(residual))


def checkpoint_rows(fitted):
    fields = fitted['checkpoint_fields'].tolist()
    rows = [dict(zip(fields, values.tolist())) for values in fitted['checkpoint_metrics']]
    for row in rows:
        for key in ('epoch', 'edges', 'lag_routes'):
            row[key] = int(row[key])
    return rows


def fit_candidate(run, segments, validation, method, profile, seed, stop_epoch=None):
    from ..shared import snapshot, state_arrays, write_npz, write_csv, read_npz
    key = dict(profile=profile, method=method, seed=seed, stop_epoch=stop_epoch,
               selection='minimum_validation_mse')
    def compute(ident):
        from gse_var.data import construct_lagged_dataset
        train_data = construct_lagged_dataset(segments, profile['order'])
        train_x, train_y = train_data.predictors, train_data.responses
        if method == 'Linear-SGL':
            out = linear_fit(segments, validation['x'], profile['group'], profile['l1'])
            predicted = validation['x'].reshape(len(validation['x']), -1)@out['beta']+out['intercept']
            train_predicted = train_x.reshape(len(train_x), -1)@out['beta']+out['intercept']
            metrics = validation_metrics(predicted, validation['y'], out['gate'], run.config['validation_blocks'])
            row = dict(epoch=0, **metrics, training_mse=float(np.mean((train_predicted-train_y)**2)))
            out.update(validation_mse=np.array(metrics['mse']), best_epoch=np.array(0), model_epoch=np.array(0),
                       validation_predictions=predicted, checkpoint_fields=np.array(CHECKPOINT_FIELDS),
                       checkpoint_metrics=np.array([[row[k] for k in CHECKPOINT_FIELDS]]))
            return out
        from gse_var import GSEVARTrainingConfig, fit_gse_var
        import torch
        last_epoch = profile['epochs'] if stop_epoch is None else int(stop_epoch)
        if not 1 <= last_epoch <= 5000:
            raise ValueError('Training must stop between epoch 1 and epoch 5000.')
        best = [float('inf'), 0]
        rows = []
        checkpoint = run.path/'best'/f'{ident}.npz'
        def observe(epoch, model, record):
            if not should_check(epoch, last_epoch, run.config):
                return
            arrays = snapshot(model, validation['x'])
            train_arrays = snapshot(model, train_x)
            metrics = validation_metrics(arrays['predictions'], validation['y'], arrays['gate'], run.config['validation_blocks'])
            row = dict(epoch=epoch, **metrics,
                       training_mse=float(np.mean((train_arrays['predictions']-train_y)**2)),
                       sampled_mse=float(record['mse']), training_loss=float(record['loss']),
                       weight_decay=float(record['weight_decay']),
                       mean_gate_probability=float(arrays['gate_probabilities'].mean()))
            rows.append(row)
            if metrics['mse'] < best[0]:
                best[:] = [metrics['mse'], epoch]
                if stop_epoch is None:
                    write_npz(checkpoint, **state_arrays(model))
            write_csv(run.path/'tables'/f'{ident}_history.csv', rows)
            run.log(f'{LABELS[method]} seed={seed} lr={profile["lr"]:g} gate_lr={profile["gate_lr"]:g} '
                    f'budget={profile["budget"]:g} epoch={epoch} train_mse={row["training_mse"]:.6g} '
                    f'validation_mse={metrics["mse"]:.6g} se={metrics["se"]:.6g} '
                    f'edges={metrics["edges"]} lag_routes={metrics["lag_routes"]} min_mse_epoch={best[1]}')
        shared = dict(order=profile['order'], hidden_layer_size=profile['hidden'],
                      num_hidden_layers=profile['layers'], max_epochs=last_epoch,
                      batch_size=sum(len(s)-profile['order'] for s in segments),
                      learning_rate=profile['lr'], gate_learning_rate=profile['gate_lr'],
                      lambda_smooth=profile['smooth'], coefficient_weight_decay=profile['decay'],
                      lambda_l0=profile['l0'], gate_probability_init=profile['probability_init'],
                      seed=seed, shuffle=False, verbose=0, device=run.device)
        fitted = fit_gse_var(segments, GSEVARTrainingConfig(**shared), epoch_callback=observe)
        if stop_epoch is None:
            weights = read_npz(checkpoint)
            model_epoch = best[1]
            fitted.model.load_state_dict({k.removeprefix('state__'): torch.as_tensor(v, device=run.device)
                                         for k, v in weights.items()})
        else:
            weights = state_arrays(fitted.model)
            model_epoch = last_epoch
        arrays = snapshot(fitted.model, validation['x'])
        restored_mse = float(np.mean((arrays['predictions']-validation['y'])**2))
        expected = best[0] if stop_epoch is None else rows[-1]['mse']
        if not np.isclose(restored_mse, expected, rtol=1e-6, atol=1e-8):
            raise RuntimeError('Saved weights do not reproduce the recorded validation MSE.')
        return dict(**weights, gate=arrays['gate'], validation_predictions=arrays['predictions'],
                    validation_mse=np.array(restored_mse), best_epoch=np.array(best[1]), model_epoch=np.array(model_epoch),
                    checkpoint_fields=np.array(CHECKPOINT_FIELDS),
                    checkpoint_metrics=np.array([[r[k] for k in CHECKPOINT_FIELDS] for r in rows]))
    return run.cached(key, compute)


def assert_reproduced(fitted, selected, validation, blocks):
    actual = validation_metrics(fitted['validation_predictions'], validation['y'], fitted['gate'], blocks)
    if (int(fitted['model_epoch']) != selected['epoch'] or
        not np.isclose(actual['mse'], selected['mse'], rtol=1e-6, atol=1e-8) or
        not np.isclose(actual['se'], selected['se'], rtol=1e-6, atol=1e-8) or
        actual['edges'] != selected['edges'] or actual['lag_routes'] != selected['lag_routes']):
        raise RuntimeError('The selected checkpoint was not reproduced; refusing to evaluate a different model.')


def execute(project, config, method, device=None):
    from ..shared import Run, digest, write_npz, write_csv, restore_model, snapshot
    if method not in METHODS:
        raise ValueError(method)
    validate_config(config)
    panel, manifest, prepared = prepare_data(project, config)
    segments, cases, processing = make_cases(panel, config)
    root = study_root(project, config, manifest)
    experiment = str(root.relative_to(Path(project)/'results')/method)
    run = Run(project, experiment, dict(config, device=device or config['devices'][method]), manifest)
    p, k = len(panel['names']), config['order']
    profiles = candidate_profiles(config, method, p)
    with run.lock():
        save_json(root/method/'current_run.json', dict(path=str(run.path.relative_to(root))))
        save_json(run.path/'array_schema.json', dict(
            coefficients=['time', 'lag_oldest_first', 'target', 'source'],
            jacobian=['time', 'lag_oldest_first', 'target', 'source'], gate=['lag_oldest_first', 'target', 'source'],
            lag_days=list(range(k, 0, -1)), checkpoint_storage='metrics for all checks; best weights per scan; selected state reproduced',
            selection='minimum validation MSE across candidate-best checkpoints'))
        write_npz(run.path/'data.npz', **panel, **processing,
                  validation_indices=cases['validation']['indices'], test_indices=cases['test']['indices'])
        save_json(run.path/'validation_reference.json', dict(
            standardized_persistence_mse=float(np.mean((cases['validation']['x'][:,-1]-cases['validation']['y'])**2))))
        candidates, all_rows = [], []
        for index, profile in enumerate(profiles):
            fitted = fit_candidate(run, segments, cases['validation'], method, profile, config['tuning_seed'])
            rows = [dict(row, candidate=index, budget=profile['budget'], profile=profile) for row in checkpoint_rows(fitted)]
            all_rows.extend(rows)
            best = min(rows, key=lambda row: (row['mse'], row['epoch']))
            candidates.append(best)
            write_csv(run.path/'validation_candidates.csv', candidates)
            write_csv(run.path/'validation_checkpoints.csv', all_rows)
        selected = choose_candidate(candidates)
        save_json(run.path/'selection.json', dict(method=method, label=LABELS[method], primary_seed=config['tuning_seed'],
                  criterion='minimum validation MSE across candidate-best checkpoints', **selected))
        run.log(f'SELECTED candidate={selected["candidate"]} epoch={selected["epoch"]} '
                f'mse={selected["mse"]:.6g} threshold={selected["threshold"]:.6g} edges={selected["edges"]}')
        seeds = config['seeds'] if method == 'GSE-VAR' else [config['tuning_seed']]
        results = []
        for seed in seeds:
            if method == 'Linear-SGL' or seed == config['tuning_seed']:
                fitted = fit_candidate(run, segments, cases['validation'], method, selected['profile'], seed)
            else:
                fitted = fit_candidate(run, segments, cases['validation'], method, selected['profile'], seed, selected['epoch'])
            if method != 'Linear-SGL' and seed == config['tuning_seed'] and int(fitted['model_epoch']) != selected['epoch']:
                fitted = fit_candidate(run, segments, cases['validation'], method, selected['profile'], seed, selected['epoch'])
            if seed == config['tuning_seed']:
                assert_reproduced(fitted, selected, cases['validation'], config['validation_blocks'])
            test = cases['test']
            if method == 'Linear-SGL':
                coefficient = fitted['beta'].reshape(k,p,p).transpose(0,2,1)
                coefficients = np.broadcast_to(coefficient, (len(test['x']),k,p,p)).copy()
                arrays = dict(predictions=test['x'].reshape(len(test['x']),-1)@fitted['beta']+fitted['intercept'],
                              coefficients=coefficients, jacobian=coefficients.copy(), gate=fitted['gate'],
                              beta=fitted['beta'], intercept=fitted['intercept'])
            else:
                model = restore_model(method, selected['profile'], p, fitted, run.device)
                arrays = snapshot(model, test['x'], jacobian=True)
            raw_prediction = np.expm1(arrays['predictions']*test['scale']+test['weekday_mean'])
            if not np.isfinite(raw_prediction).all():
                raise FloatingPointError('Inverse-transformed predictions are not finite.')
            raw_prediction = np.maximum(raw_prediction,0)
            metrics = dict(method=LABELS[method], seed=seed, validation_mse=float(fitted['validation_mse']),
                selected_epoch=int(fitted['model_epoch']), minimum_validation_mse=selected['minimum_validation_mse'],
                validation_threshold=selected['threshold'],
                test_mse=float(np.mean((arrays['predictions']-test['y'])**2)),
                test_mae=float(np.mean(np.abs(arrays['predictions']-test['y']))),
                test_rmsle=float(np.sqrt(np.mean((np.log1p(raw_prediction)-np.log1p(test['raw']))**2))),
                nonself_edges=int(((arrays['gate'] != 0).any(0)&~np.eye(p,dtype=bool)).sum()))
            weights = {name:value for name,value in fitted.items() if name.startswith('state__')}
            write_npz(run.path/'evaluation'/f'seed_{seed}.npz', **arrays, **weights, dates=test['dates'],
                      observed=test['y'], raw_observed=test['raw'], raw_predictions=raw_prediction,
                      names=panel['names'], labels=panel['labels'], selected_epoch=fitted['model_epoch'])
            results.append(metrics)
        write_csv(run.path/'evaluation_metrics.csv', results)
        save_json(run.path/'completed.json', dict(method=method, study=root.name, seeds=seeds,
                   selection_digest=digest(selected), prepared=str(prepared.relative_to(project))))
    return run.path
