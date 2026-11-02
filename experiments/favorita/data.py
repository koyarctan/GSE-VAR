"""Calendar-aligned category sales from the official Store Sales release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

SOURCE = 'https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data'
FAMILIES = (
    ('BEVERAGES', '飲料'), ('BREAD/BAKERY', 'パン・ベーカリー'),
    ('DAIRY', '乳製品'), ('DELI', 'デリ食品'), ('EGGS', '卵'),
    ('FROZEN FOODS', '冷凍食品'), ('LIQUOR,WINE,BEER', '酒類'),
    ('MEATS', '肉類'), ('POULTRY', '家禽肉'),
    ('PREPARED FOODS', '調理済食品'), ('PRODUCE', '青果'), ('SEAFOOD', '魚介'),
)


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise TypeError('Object arrays are not supported.')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def _read_source(raw_dir):
    """Read only two known members; never extract untrusted archive paths."""
    raw_dir = Path(raw_dir)
    archive = raw_dir / 'store-sales-time-series-forecasting.zip'
    if (raw_dir / 'train.csv').is_file() and (raw_dir / 'stores.csv').is_file():
        paths = [raw_dir / name for name in ('train.csv', 'stores.csv')]
        return pd.read_csv(paths[0], chunksize=250000), pd.read_csv(paths[1]), {
            p.name: fingerprint(p) for p in paths}
    if archive.is_file():
        hashes = {archive.name: fingerprint(archive)}
        def chunks():
            with zipfile.ZipFile(archive) as zipped:
                with zipped.open('train.csv') as stream:
                    yield from pd.read_csv(stream, chunksize=250000)
        with zipfile.ZipFile(archive) as zipped:
            with zipped.open('stores.csv') as stream:
                stores = pd.read_csv(stream)
        return chunks(), stores, hashes
    raise FileNotFoundError(
        f'{raw_dir} に公式のカテゴリ版 train.csv と stores.csv、または '
        'store-sales-time-series-forecasting.zip を配置してください。'
        f'取得先: {SOURCE}')


def prepare_data(project, config):
    directory = Path(project) / 'data' / 'favorita'
    chunks, stores, hashes = _read_source(directory / 'raw')
    settings = {k: config[k] for k in ('start', 'train_end', 'validation_end', 'end',
                                       'min_nonzero_fraction', 'order')}
    spec = dict(schema=1, settings=settings, families=FAMILIES, raw_sha256=hashes, source=SOURCE)
    prepared = directory / 'prepared' / identity(spec)[:16]
    if (prepared / 'manifest.json').is_file() and (prepared / 'panel.npz').is_file():
        # Close a possible pandas reader without consuming the raw file again.
        if hasattr(chunks, 'close'):
            chunks.close()
        return load_npz(prepared / 'panel.npz'), json.loads((prepared / 'manifest.json').read_text(encoding='utf-8')), prepared
    names = [v[0] for v in FAMILIES]
    pieces, market_dates = [], set()
    for chunk in chunks:
        required = {'date', 'store_nbr', 'family', 'sales'}
        if not required.issubset(chunk):
            raise ValueError('カテゴリ版 Store Sales のCSVが必要です。SKU版 unit_sales は入力できません。')
        within = chunk['date'].between(config['start'], config['end'])
        market_dates.update(chunk.loc[within, 'date'].astype(str))
        selected = chunk.loc[within & chunk['family'].isin(names), ['date', 'store_nbr', 'family', 'sales']].copy()
        if not np.isfinite(selected['sales']).all() or (selected['sales'] < 0).any():
            raise ValueError('sales に負値または非有限値があります。定義を確認してください。')
        pieces.append(selected)
    sales = pd.concat(pieces, ignore_index=True)
    if sales.duplicated(['date', 'store_nbr', 'family']).any():
        raise ValueError('同一日・店舗・カテゴリが重複しています。')
    absent = sorted(set(names) - set(sales['family']))
    if absent:
        raise ValueError(f'事前指定カテゴリが不足しています: {absent}')
    dates = pd.date_range(config['start'], config['end'], freq='D')
    if not config['start'] < config['train_end'] < config['validation_end'] < config['end']:
        raise ValueError('期間の順序が不正です。')
    training_days = sorted(d for d in market_dates if d <= config['train_end'])
    if len(training_days) < 200:
        raise ValueError('訓練期間の観測日が200日未満です。')
    quality, panels = [], {}
    for store_id, frame in sales.groupby('store_nbr', sort=True):
        wide = frame.pivot(index='date', columns='family', values='sales').reindex(columns=names)
        train = wide.reindex(training_days)
        present = not train.isna().any().any()
        nonzero = float((train > 0).mean().min())
        varying = bool((train.std() > 0).all())
        eligible = present and varying and nonzero >= config['min_nonzero_fraction']
        quality.append(dict(store_nbr=int(store_id), complete_training=present,
                            min_nonzero_fraction=nonzero, positive_variance=varying, eligible=eligible))
        if eligible:
            panels[int(store_id)] = wide
    prepared.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(quality).to_csv(prepared / 'store_quality.csv', index=False, encoding='utf-8-sig')
    if not panels:
        raise ValueError(f'訓練期間の品質条件を満たす店舗がありません。{prepared / "store_quality.csv"} を確認してください。')
    store_id = min(panels)
    wide = panels[store_id].reindex(dates.strftime('%Y-%m-%d'))
    expected = wide.index.isin(market_dates)
    if wide.loc[expected].isna().any().any():
        raise ValueError('選定店舗の検証・評価期間にカテゴリ単位の欠測があります。店舗を変更せず入力を確認してください。')
    values = wide.to_numpy(dtype=np.float64)
    panel = dict(raw=values, dates=dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
                 names=np.array(names), labels=np.array([v[1] for v in FAMILIES]))
    store_row = stores.loc[stores['store_nbr'] == store_id]
    if len(store_row) != 1:
        raise ValueError('stores.csv で選定店舗を一意に識別できません。')
    store_metadata = json.loads(store_row.to_json(orient='records'))[0]
    missing_days = panel['dates'][~expected].tolist()
    manifest = dict(spec)
    manifest.update(prepared_id=identity(spec)[:16], store=store_metadata,
                    selection='smallest eligible store ID; training data only',
                    missing_calendar_days=missing_days, calendar_rows=len(values),
                    available_days=int(expected.sum()), families=[dict(family=a, label=b) for a, b in FAMILIES])
    save_npz(prepared / 'panel.npz', **panel)
    save_json(prepared / 'manifest.json', manifest)
    wide.to_csv(prepared / 'category_sales.csv', index_label='date', encoding='utf-8-sig')
    pd.DataFrame(manifest['families']).to_csv(prepared / 'category_dictionary.csv', index=False, encoding='utf-8-sig')
    return panel, manifest, prepared


def make_cases(panel, config):
    """Fit preprocessing on training observations; split at calendar gaps."""
    dates = pd.to_datetime(panel['dates'])
    train = dates <= config['train_end']
    known = np.isfinite(panel['raw']).all(axis=1)
    if not np.array_equal(np.isfinite(panel['raw']).any(axis=1), known):
        raise ValueError('一部のカテゴリだけが欠測している日があります。')
    logs = np.log1p(panel['raw'])
    weekday = dates.day_of_week.to_numpy()
    means = np.stack([logs[train & known & (weekday == day)].mean(0) for day in range(7)])
    residual = logs - means[weekday]
    scale = residual[train & known].std(0, ddof=0)
    if not np.isfinite(means).all() or not np.isfinite(scale).all() or (scale < 1e-8).any():
        raise ValueError('曜日調整・標準化を訓練期間で推定できません。')
    z = (residual / scale).astype(np.float32)
    k = config['order']
    valid_train = train & known
    boundaries = np.flatnonzero(np.diff(np.r_[False, valid_train, False]))
    segments = [z[a:b].copy() for a, b in zip(boundaries[::2], boundaries[1::2]) if b-a > k]
    if not segments:
        raise ValueError('連続した訓練区間が不足しています。')
    cases = {}
    valid_windows = [t for t in range(k, len(z)) if known[t-k:t+1].all()]
    for name, low, high in [('validation', config['train_end'], config['validation_end']),
                            ('test', config['validation_end'], config['end'])]:
        inds = np.array([t for t in valid_windows if low < panel['dates'][t] <= high], dtype=int)
        if len(inds) < 28:
            raise ValueError(f'{name} の有効な日次ラグ窓が28個未満です。')
        cases[name] = dict(x=np.stack([z[t-k:t] for t in inds]), y=z[inds], indices=inds,
                           dates=panel['dates'][inds], raw=panel['raw'][inds],
                           weekday_mean=means[weekday[inds]], scale=scale)
    return segments, cases, dict(z=z, weekday_mean=means, scale=scale,
                                  known=known, train=train, training_samples=sum(len(s)-k for s in segments))
