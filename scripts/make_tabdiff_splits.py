"""Build the SECOND, TabDiff-comparable split for the TabDiff-paper datasets.

The original splits in data/<dataset>/ are left untouched — they remain the
ones used for the CatBoost/TabDDPM-protocol evaluation. This script re-splits
the SAME rows (concat of the existing train/val/test arrays) into new
partitions matching TabDiff's Table 6 sizes and writes them to
data/<dataset>_tabdiff_split/ with the identical npy layout, so train.py /
sample.py / eval_xgboost.py can point real_data_path at the new folder
unchanged.

Method (fixed, reproducible):
  * sklearn.model_selection.train_test_split with SEED = 0
  * stratified on the label for classification, unstratified for regression
  * exact row counts: the test partition is carved out first, then val
  * adult: the existing 16,281-row test split is the OFFICIAL adult test set
    and already matches Table 6 exactly — it is kept as-is; only the remaining
    32,561 rows are re-split into 28,943 train / 3,618 val.
  * magic: TabDiff's Table 6 total (19,019) is one short of the full dataset
    (19,020); the 1 leftover row is dropped to match the published counts.
  * beijing: Table 6 (35,058/4,383/4,383 = 43,824) describes the RAW file
    including 2,067 rows with missing pm2.5 — TabDiff's own preprocess_beijing
    drops those rows, so no NaN-free beijing of that size exists. We instead
    apply TabDiff's split PROCEDURE (90/10 train/test, val = train/9) to the
    41,757 clean rows: 33,406 / 4,175 / 4,176.
  * diabetes: skipped — data/diabetes is Pima, not TabDiff's CDC dataset;
    resolve the dataset-identity issue before splitting.
"""

import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

SEED = 0
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data')

# dataset -> (n_train, n_val, n_test, keep_original_test)
TABDIFF_SPLITS = {
    'adult':    (28943, 3618, 16281, True),
    'default':  (24000, 3000, 3000,  False),
    'shoppers': (9864,  1233, 1233,  False),
    'magic':    (15215, 1902, 1902,  False),
    'beijing':  (33406, 4175, 4176,  False),  # see module docstring
    'news':     (31714, 3965, 3965,  False),
}

SPLITS = ('train', 'val', 'test')


def _load_part(src, split):
    def _opt(name):
        p = os.path.join(src, f'{name}_{split}.npy')
        return np.load(p, allow_pickle=True) if os.path.exists(p) else None
    return _opt('X_num'), _opt('X_cat'), np.load(os.path.join(src, f'y_{split}.npy'),
                                                 allow_pickle=True)


def _concat(parts):
    if any(p is None for p in parts):
        return None
    return np.concatenate(parts, axis=0)


def make_split(ds):
    n_train, n_val, n_test, keep_test = TABDIFF_SPLITS[ds]
    src = os.path.join(DATA, ds)
    dst = os.path.join(DATA, f'{ds}_tabdiff_split')
    os.makedirs(dst, exist_ok=True)

    info = json.load(open(os.path.join(src, 'info.json')))
    task = info['task_type']
    stratify_ok = task != 'regression'

    loaded = {s: _load_part(src, s) for s in SPLITS}

    if keep_test:
        pool_splits = ('train', 'val')
        test_arrays = loaded['test']
        assert len(test_arrays[2]) == n_test, \
            f'{ds}: existing test size {len(test_arrays[2])} != Table 6 {n_test}'
    else:
        pool_splits = SPLITS
        test_arrays = None

    X_num = _concat([loaded[s][0] for s in pool_splits])
    X_cat = _concat([loaded[s][1] for s in pool_splits])
    y = np.concatenate([loaded[s][2] for s in pool_splits], axis=0)

    n_pool = len(y)
    needed = n_train + n_val + (0 if keep_test else n_test)
    assert needed <= n_pool, f'{ds}: need {needed} rows, pool has {n_pool}'
    n_leftover = n_pool - needed

    idx = np.arange(n_pool)
    strat = y if stratify_ok else None

    if keep_test:
        idx_rest = idx
    else:
        idx_rest, idx_test = train_test_split(
            idx, test_size=n_test, random_state=SEED, stratify=strat)

    strat_rest = y[idx_rest] if stratify_ok else None
    idx_rest2, idx_val = train_test_split(
        idx_rest, test_size=n_val, random_state=SEED, stratify=strat_rest)

    if n_leftover:
        # A leftover this small cannot be carved stratified (magic drops a
        # single row); a seeded shuffle keeps it reproducible.
        rng = np.random.RandomState(SEED)
        shuffled = rng.permutation(idx_rest2)
        idx_train, _dropped = shuffled[:n_train], shuffled[n_train:]
        print(f'  [{ds}] dropped {len(_dropped)} leftover row(s) to match '
              f'the published Table 6 total')
    else:
        idx_train = idx_rest2

    new = {
        'train': (X_num[idx_train] if X_num is not None else None,
                  X_cat[idx_train] if X_cat is not None else None, y[idx_train],
                  idx_train),
        'val':   (X_num[idx_val] if X_num is not None else None,
                  X_cat[idx_val] if X_cat is not None else None, y[idx_val],
                  idx_val),
    }
    if keep_test:
        new['test'] = (*test_arrays, np.array([], dtype=int))
    else:
        new['test'] = (X_num[idx_test] if X_num is not None else None,
                       X_cat[idx_test] if X_cat is not None else None, y[idx_test],
                       idx_test)

    for split, (xn, xc, yy, ids) in new.items():
        if xn is not None:
            np.save(os.path.join(dst, f'X_num_{split}.npy'), xn)
        if xc is not None:
            np.save(os.path.join(dst, f'X_cat_{split}.npy'), xc)
        np.save(os.path.join(dst, f'y_{split}.npy'), yy)
        np.save(os.path.join(dst, f'idx_{split}.npy'), ids)

    new_info = dict(info)
    new_info['name'] = f"{info.get('name', ds)} (TabDiff split)"
    new_info['id'] = f'{ds}--tabdiff-split'
    new_info['train_size'] = int(len(new['train'][2]))
    new_info['val_size'] = int(len(new['val'][2]))
    new_info['test_size'] = int(len(new['test'][2]))
    new_info['split_provenance'] = {
        'description': 'Second split matching TabDiff Table 6 sizes; the '
                       'original split in data/' + ds + '/ is unchanged and '
                       'remains the CatBoost/TabDDPM-protocol split.',
        'method': 'sklearn train_test_split, test first then val, '
                  + ('stratified' if stratify_ok else 'unstratified'),
        'seed': SEED,
        'source': f'data/{ds}/ (concat of original train/val/test rows)',
        'kept_original_test': keep_test,
    }
    with open(os.path.join(dst, 'info.json'), 'w') as fh:
        json.dump(new_info, fh, indent=4)

    sizes = tuple(len(new[s][2]) for s in SPLITS)
    print(f'  [{ds}] {sizes[0]}/{sizes[1]}/{sizes[2]}  -> {dst}')
    if stratify_ok:
        for s in SPLITS:
            vals, counts = np.unique(new[s][2], return_counts=True)
            frac = {int(v): round(c / counts.sum(), 4) for v, c in zip(vals, counts)}
            print(f'      {s:5s} class balance: {frac}')
    return sizes


if __name__ == '__main__':
    for ds in TABDIFF_SPLITS:
        make_split(ds)
    print('\nAll TabDiff splits written. Original data/<dataset>/ folders untouched.')
