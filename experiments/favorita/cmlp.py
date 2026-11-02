"""Favorita cMLP: proximal training and minimum-validation-MSE selection."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import sys
import zipfile

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np


from ..shared import write_json, write_csv, write_npz, read_npz


def inverse(pred, dates, data):
    weekday = (np.asarray(dates, dtype='datetime64[D]').astype(np.int64) + 3) % 7
    with np.errstate(over='raise', invalid='raise'):
        raw = np.expm1(np.asarray(pred, float) * data['scale'] + data['weekday_mean'][weekday])
    return np.maximum(raw, 0.)


def prepare(data, metadata):
    k = int(metadata['order']); z = data['z']; n, p = z.shape
    valid = np.array([t for t in range(k, n) if data['known'][t-k:t+1].all()], dtype=int)
    train_indices = np.array([t for t in valid if data['train'][t-k:t+1].all()], dtype=int)
    parts = {'train': train_indices, 'validation': data['validation_indices'], 'test': data['test_indices']}
    for name, ids in parts.items():
        assert len(ids) and len(np.unique(ids)) == len(ids), name
        assert np.isin(ids, valid).all(), f'{name}: invalid lag window'
        assert np.isfinite(z[np.concatenate([np.arange(t-k, t+1) for t in ids])]).all(), name
    assert train_indices[-1] < parts['validation'][0] < parts['validation'][-1] < parts['test'][0]
    assert len(train_indices) == int(data['training_samples'])
    mask = data['train'] & data['known']
    cuts = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    segments = [z[a:b].copy() for a, b in zip(cuts[::2], cuts[1::2]) if b-a > k]
    assert sum(len(v)-k for v in segments) == len(train_indices)
    cases = {name: {'x': np.stack([z[t-k:t] for t in ids]), 'y': z[ids], 'indices': ids,
                    'dates': data['dates'][ids]} for name, ids in parts.items()}
    # Verify baseline predictions when frozen comparison arrays are supplied.
    for method in ('GSE-DisARM', 'Linear-SGL'):
        if f'baseline__{method}__predictions' not in data: continue
        pred = data[f'baseline__{method}__predictions']
        raw = inverse(pred, cases['test']['dates'], data)
        assert np.allclose(raw, data[f'baseline__{method}__raw_predictions'], rtol=1e-6, atol=1e-4)
        assert np.allclose(data['raw'][parts['test']], data[f'baseline__{method}__raw_observed'])
    return segments, cases


def checkpoints(epochs, every):
    return sorted({1, epochs, *range(20, min(epochs, 200)+1, 20), *range(every, epochs+1, every)})


def select(rows):
    if not rows or not all(np.isfinite(r['validation_mse']) for r in rows):
        raise ValueError('Candidate validation MSE must be finite.')
    return min(rows, key=lambda r: (r['validation_mse'], r['best_epoch'], r['candidate']))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[2]/'data/favorita/cmlp_inputs.npz')
    p.add_argument('--device', default=os.environ.get('GSE_DEVICE', 'cuda:0'))
    p.add_argument('--epochs', type=int, default=5000)
    p.add_argument('--hidden', type=int, default=100)
    p.add_argument('--lambdas', type=float, nargs='+', default=[0, .001, .003, .01, .03, .1, .3, .5, 1, 10])
    p.add_argument('--lrs', type=float, nargs='+', default=[.01, .003])
    p.add_argument('--ridge', type=float, default=.01)
    p.add_argument('--check-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--check-data', action='store_true')
    p.add_argument('--smoke', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.smoke:
        args.epochs, args.hidden, args.lambdas, args.lrs, args.check_every = 3, 4, [0., .1], [.01], 1
    if not (1 <= args.epochs <= 5000 and args.hidden > 0 and args.check_every > 0):
        raise ValueError('epochs must be 1..5000; hidden and check-every must be positive.')
    if min(args.lambdas) < 0 or min(args.lrs) <= 0 or args.ridge < 0:
        raise ValueError('Invalid lambda, learning rate, or ridge penalty.')
    args.project = args.project.resolve(); args.data = args.data.resolve()
    if not args.data.exists():
        from .run import configuration
        from .data import prepare_data,make_cases
        config=configuration()
        panel,manifest,_=prepare_data(args.project,config)
        _,cases,processing=make_cases(panel,config)
        write_npz(args.data,**panel,**processing,
                  validation_indices=cases['validation']['indices'],test_indices=cases['test']['indices'])
        write_json(args.data.with_suffix('.json'),dict(order=config['order'],
            input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
            baseline_metrics=[],data=manifest))
    metadata = json.loads(args.data.with_suffix('.json').read_text(encoding='utf-8'))
    data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if data_sha != metadata['input_sha256']:
        raise ValueError('Input file checksum does not match the frozen Favorita data.')
    data = read_npz(args.data)
    segments, cases = prepare(data, metadata)
    print('DATA', json.dumps({k: len(v['y']) for k, v in cases.items()}), flush=True)
    if args.check_data:
        print('Frozen data, split indices, missing-day exclusions and inverse transforms verified.', flush=True)
        return
    sys.path.insert(0, str(args.project))
    import torch
    from gse_var import CMLP, CMLPTrainingConfig, fit_cmlp
    from gse_var.data import construct_lagged_dataset
    import gse_var.cmlp as cmlp_module
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the selected Python environment.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    k, p = int(metadata['order']), data['z'].shape[1]
    train_data = construct_lagged_dataset(segments, k)
    assert np.array_equal(train_data.predictors, cases['train']['x'])
    assert np.array_equal(train_data.responses, cases['train']['y'])
    model_source = Path(cmlp_module.__file__)
    source_sha = hashlib.sha256(model_source.read_bytes()).hexdigest()
    spec = dict(order=k, hidden=args.hidden, layers=1, epochs=args.epochs, ridge=args.ridge,
                lambdas=args.lambdas, learning_rates=args.lrs, check_every=args.check_every,
                early_check_every=20, early_check_until=200, seed=args.seed,
                regularizer='hierarchical_group_lasso', optimizer='ista', data_sha256=data_sha,
                cmlp_source_sha256=source_sha, criterion='minimum validation MSE', smoke=args.smoke)
    ident = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    out = args.project/'results/11_Favorita_cMLP'/('smoke_' if args.smoke else 'full_')
    out = out.with_name(out.name+ident); out.mkdir(parents=True, exist_ok=True)
    write_json(out/'manifest.json', dict(config=spec, data=metadata, device=args.device,
               environment=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None)))
    (out/'source_cmlp.py').write_bytes(model_source.read_bytes())
    (out/'run_favorita_cmlp.py').write_bytes(Path(__file__).read_bytes())
    print('OUTPUT', out, flush=True)
    tensors = {name: (torch.as_tensor(v['x'], device=args.device), torch.as_tensor(v['y'], device=args.device)) for name, v in cases.items() if name != 'test'}
    def restore(model, arrays, prefix):
        model.load_state_dict({key[len(prefix):]: torch.as_tensor(value, device=args.device)
                               for key, value in arrays.items() if key.startswith(prefix)})
    def states(model, prefix):
        return {prefix+key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}
    def predictions(model, name):
        model.eval()
        with torch.no_grad(): return model(tensors[name][0]).cpu().numpy()
    candidates = []
    history_fields = ['epoch', 'training_mse', 'validation_mse', 'edges', 'lag_routes']
    for number, (lr, lam) in enumerate(itertools.product(args.lrs, args.lambdas)):
        folder = out/'candidates'/f'{number:02d}'; folder.mkdir(parents=True, exist_ok=True)
        path = folder/'checkpoint.npz'
        torch.manual_seed(args.seed)
        model = CMLP(p, k, args.hidden, 1).to(args.device)
        epoch, best_epoch, best_mse, history, best_state = 0, 0, float('inf'), [], {}
        if path.exists():
            saved = read_npz(path); restore(model, saved, 'last_state__')
            epoch, best_epoch, best_mse = int(saved['epoch']), int(saved['best_epoch']), float(saved['best_mse'])
            history = [dict(zip(history_fields, row.tolist())) for row in saved['history']]
            best_state = {key: value for key, value in saved.items() if key.startswith('best_state__')}
            print(f'RESUME candidate={number} epoch={epoch}', flush=True)
        for stop in checkpoints(args.epochs, args.check_every):
            if stop <= epoch: continue
            cfg = CMLPTrainingConfig(order=k, hidden_layer_size=args.hidden, num_hidden_layers=1,
                      max_epochs=stop-epoch, batch_size=len(cases['train']['y']), learning_rate=lr,
                      lambda_ngc=lam, ridge_lambda=args.ridge, regularizer='hierarchical_group_lasso',
                      optimizer='ista', seed=args.seed, shuffle=False, causal_threshold=0., verbose=0, device=args.device)
            # Plain SGD has no momentum state; consecutive full-batch chunks preserve its update sequence.
            model = fit_cmlp(segments, cfg, model=model).model
            if not all(torch.isfinite(v).all().item() for v in model.parameters()):
                raise FloatingPointError(f'Nonfinite weights: candidate={number}, epoch={stop}')
            val_pred = predictions(model, 'validation'); train_pred = predictions(model, 'train')
            val_mse = float(np.mean((val_pred-cases['validation']['y'])**2))
            train_mse = float(np.mean((train_pred-cases['train']['y'])**2))
            if not np.isfinite(val_mse): raise FloatingPointError('Nonfinite validation MSE.')
            lag_strength = model.gc_strength(False).detach().cpu().numpy().transpose(2, 0, 1)
            support = lag_strength > 0; offdiag = ~np.eye(p, dtype=bool)
            row = dict(epoch=stop, training_mse=train_mse, validation_mse=val_mse,
                       edges=int(support.any(0)[offdiag].sum()), lag_routes=int(support[:, offdiag].sum()))
            history.append(row); epoch = stop
            if val_mse < best_mse:
                best_mse, best_epoch, best_state = val_mse, stop, states(model, 'best_state__')
            write_npz(path, **states(model, 'last_state__'), **best_state, epoch=np.array(epoch),
                      best_epoch=np.array(best_epoch), best_mse=np.array(best_mse),
                      history=np.array([[r[f] for f in history_fields] for r in history]), history_fields=np.array(history_fields))
            write_csv(folder/'history.csv', history)
            print(f'cMLP candidate={number+1}/{len(args.lrs)*len(args.lambdas)} lambda={lam:g} lr={lr:g} epoch={epoch} '
                  f'validation_mse={val_mse:.7g} edges={row["edges"]} best_epoch={best_epoch}', flush=True)
        selected_row = next(r for r in history if int(r['epoch']) == best_epoch)
        candidates.append(dict(candidate=number, lambda_ngc=lam, learning_rate=lr, best_epoch=best_epoch,
                               validation_mse=best_mse, edges=int(selected_row['edges']), lag_routes=int(selected_row['lag_routes'])))
        write_csv(out/'validation_candidates.csv', candidates)
    # All candidates and checkpoint choices are fixed before evaluating test predictions.
    chosen = select(candidates)
    saved = read_npz(out/'candidates'/f'{chosen["candidate"]:02d}'/'checkpoint.npz')
    restore(model, saved, 'best_state__')
    val_pred = predictions(model, 'validation')
    assert np.isclose(np.mean((val_pred-cases['validation']['y'])**2), chosen['validation_mse'], rtol=1e-6, atol=1e-8)
    model.eval()
    test_x = torch.as_tensor(cases['test']['x'], device=args.device)
    with torch.no_grad(): pred = model(test_x).cpu().numpy()
    strength = model.gc_strength(True).detach().cpu().numpy()
    lag_strength = model.gc_strength(False).detach().cpu().numpy().transpose(2, 0, 1)
    graph = (strength > 0).astype(np.int8); offdiag = ~np.eye(p, dtype=bool)
    raw_pred = inverse(pred, cases['test']['dates'], data); raw_obs = data['raw'][cases['test']['indices']]
    jac_input = test_x.detach().requires_grad_(True); jac_pred = model(jac_input)
    jac = torch.stack([torch.autograd.grad(jac_pred[:, i].sum(), jac_input, retain_graph=i < p-1)[0]
                       for i in range(p)], dim=2).detach().cpu().numpy()
    write_npz(out/'cMLP_evaluation.npz', predictions=pred, observed=cases['test']['y'], raw_predictions=raw_pred,
              raw_observed=raw_obs, validation_predictions=val_pred, validation_observed=cases['validation']['y'],
              dates=cases['test']['dates'], validation_dates=cases['validation']['dates'], names=data['names'], labels=data['labels'],
              graph=graph, strength=strength, lag_strength=lag_strength, lag_support=(lag_strength > 0).astype(np.int8),
              jacobian=jac, selected_epoch=np.array(chosen['best_epoch']), **states(model, 'state__'))
    selection = dict(chosen, method='cMLP', criterion='minimum validation MSE across candidate-best checkpoints',
                     test_used_for_selection=False, order=k, hidden=args.hidden, ridge=args.ridge, seed=args.seed,
                     data_sha256=data_sha, cmlp_source_sha256=source_sha)
    write_json(out/'selection.json', selection)
    edges = [dict(target=str(data['names'][i]), source=str(data['names'][j]), target_label=str(data['labels'][i]),
                  source_label=str(data['labels'][j]), selected=int(graph[i,j]), strength=float(strength[i,j]), self_edge=int(i == j))
             for i in range(p) for j in range(p)]
    write_csv(out/'cMLP_edges.csv', edges)
    routes = [dict(target=str(data['names'][i]), source=str(data['names'][j]), lag_days=k-l,
                   selected=int(lag_strength[l,i,j] > 0), strength=float(lag_strength[l,i,j]), self_edge=int(i == j))
              for l in range(k) for i in range(p) for j in range(p)]
    write_csv(out/'cMLP_lag_routes.csv', routes)
    metrics = [dict(method='cMLP', seed=args.seed, validation_mse=chosen['validation_mse'],
                    selected_epoch=chosen['best_epoch'], nonself_edges=int(graph[offdiag].sum()),
                    test_mse=float(np.mean((pred-cases['test']['y'])**2)),
                    test_rmse=float(np.sqrt(np.mean((raw_pred-raw_obs)**2))),
                    test_mae=float(np.mean(np.abs(raw_pred-raw_obs))), rmse_unit='daily sales quantity')]
    write_csv(out/'evaluation_metrics.csv', metrics)
    comparison = metrics + metadata['baseline_metrics']
    fields = list(metrics[0])
    comparison = [{key: row.get(key, '') for key in fields} for row in comparison]
    write_csv(out/'comparison_metrics.csv', comparison)
    write_json(out/'completion.json', dict(complete=True, expected=len(candidates), selected=selection))
    archive = out.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['manifest.json','selection.json','validation_candidates.csv','cMLP_evaluation.npz',
                     'cMLP_edges.csv','cMLP_lag_routes.csv','evaluation_metrics.csv','comparison_metrics.csv',
                     'completion.json','source_cmlp.py','run_favorita_cmlp.py']:
            z.write(out/name, name)
        z.write(args.data, 'data.npz'); z.write(args.data.with_suffix('.json'), 'data_metadata.json')
        for path in sorted((out/'candidates').glob('*/history.csv')):
            z.write(path, path.relative_to(out).as_posix())
    print('SELECTED', json.dumps(selection, ensure_ascii=False), flush=True)
    print('METRICS', json.dumps(metrics[0], ensure_ascii=False), flush=True)
    print('RESULT_ZIP', archive, flush=True)
    return out


if __name__ == '__main__':
    main()
