"""Retype beijing/news *_tabdiff_split features to TabDiff's official schema.

Applies ONLY to data/beijing_tabdiff_split/ and data/news_tabdiff_split/ —
the original data/beijing/ and data/news/ (CatBoost/TabDDPM protocol) are
never touched. The train/val/test ROW assignment built by
make_tabdiff_splits.py is preserved exactly (the saved idx_*.npy files are
reused); only the column typing/encoding changes.

Schema source: process_dataset.py + data/Info/{beijing,news}.json from
MinkaiXu/TabDiff (read directly from their repo, not reconstructed):

beijing (preprocess_beijing drops the 'No' column; Info/beijing.json):
    categorical: year, month, day, hour, cbwd          (was: cbwd only)
    numerical:   DEWP, TEMP, PRES, Iws, Is, Ir         (was: + year..hour)
    target:      pm2.5

news (preprocess_news drops only 'url', keeps timedelta):
    numerical:   timedelta + the 44 existing numerical columns (45 total)
    categorical: data_channel = argmax over the 6 data_channel_is_* dummies
                     (order: lifestyle, entertainment, bus, socmed, tech, world)
                 weekday      = argmax over weekday_is_monday..sunday + is_weekend
                     (np.argmax first-max semantics, exactly as TabDiff)
    target:      shares
    The 14 dummy columns are removed. timedelta is not present in data/news/
    (the original pipeline dropped it), so it is re-attached from the raw
    OnlineNewsPopularity.csv via the saved raw-row indices in
    data/news/idx_*.npy. Alignment against the existing arrays is verified
    before anything is written.

Raw file provenance: data/raw/OnlineNewsPopularity.csv is committed to the
repo because the UCI archive was unreachable when timedelta had to be
restored. It was downloaded from the GitHub mirror
  https://raw.githubusercontent.com/susobhang70/OnlineNewsPopularity/master/OnlineNewsPopularity.csv
(sha1 4175a1969e9c9a771da416d27ba735cb8d8e0525) and content-verified against
the processed data/news/ arrays before use: all 44 numerical columns plus the
'shares' target match exactly at the saved idx_*.npy raw-row indices, i.e. it
is the same file the pipeline was originally processed from.

Run:  python scripts/apply_tabdiff_schema.py
      (uses data/raw/OnlineNewsPopularity.csv; override with --news_raw)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, 'data')
SPLITS = ('train', 'val', 'test')

CHANNEL_DUMMIES = ['data_channel_is_lifestyle', 'data_channel_is_entertainment',
                   'data_channel_is_bus', 'data_channel_is_socmed',
                   'data_channel_is_tech', 'data_channel_is_world']
WEEKDAY_DUMMIES = ['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday',
                   'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday',
                   'weekday_is_sunday', 'is_weekend']


def _load_pool(ds):
    """Original data/<ds>/ rows in pool order (train,val,test) — the order the
    tabdiff-split idx files index into."""
    src = os.path.join(DATA, ds)
    X_num = np.concatenate([np.load(os.path.join(src, f'X_num_{s}.npy'),
                                    allow_pickle=True) for s in SPLITS])
    X_cat = np.concatenate([np.load(os.path.join(src, f'X_cat_{s}.npy'),
                                    allow_pickle=True) for s in SPLITS])
    y = np.concatenate([np.load(os.path.join(src, f'y_{s}.npy'),
                                allow_pickle=True) for s in SPLITS])
    return X_num, X_cat, y


def _write_split(dst, split, X_num, X_cat, y, rows):
    np.save(os.path.join(dst, f'X_num_{split}.npy'), X_num[rows])
    np.save(os.path.join(dst, f'X_cat_{split}.npy'), X_cat[rows])
    np.save(os.path.join(dst, f'y_{split}.npy'), y[rows])


def _update_info(dst, num_cols, cat_cols, note):
    path = os.path.join(dst, 'info.json')
    info = json.load(open(path))
    info['n_num_features'] = len(num_cols)
    info['n_cat_features'] = len(cat_cols)
    info['num_col_names'] = num_cols
    info['cat_col_names'] = cat_cols
    info['tabdiff_schema'] = note
    with open(path, 'w') as fh:
        json.dump(info, fh, indent=4)


def apply_beijing():
    ds, dst = 'beijing', os.path.join(DATA, 'beijing_tabdiff_split')
    X_num, X_cat, y = _load_pool(ds)
    orig_num = json.load(open(os.path.join(DATA, ds, 'info.json')))['num_col_names']
    assert orig_num == ['year', 'month', 'day', 'hour',
                       'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir'], orig_num

    new_num = X_num[:, 4:10].astype(np.float32)          # DEWP..Ir
    ymdh = X_num[:, 0:4].astype(np.int64).astype(np.str_)  # year..hour as categories
    new_cat = np.column_stack([ymdh, X_cat.astype(np.str_)])  # + cbwd

    for split in SPLITS:
        rows = np.load(os.path.join(dst, f'idx_{split}.npy'))
        old_y = np.load(os.path.join(dst, f'y_{split}.npy'), allow_pickle=True)
        assert np.array_equal(old_y, y[rows]), f'beijing {split}: row assignment drifted'
        _write_split(dst, split, new_num, new_cat, y, rows)

    _update_info(dst,
                 ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir'],
                 ['year', 'month', 'day', 'hour', 'cbwd'],
                 'Feature typing matches TabDiff Info/beijing.json: year/month/'
                 'day/hour retyped numerical->categorical. Rows unchanged.')
    print('  [beijing] 6 num / 5 cat written (was 10 num / 1 cat); rows unchanged')


def apply_news(raw_csv):
    ds, dst = 'news', os.path.join(DATA, 'news_tabdiff_split')
    X_num, X_cat, y = _load_pool(ds)
    info_src = json.load(open(os.path.join(DATA, ds, 'info.json')))
    assert info_src['cat_col_names'] == CHANNEL_DUMMIES + WEEKDAY_DUMMIES, \
        'unexpected dummy-column order in data/news'

    raw = pd.read_csv(raw_csv)
    raw.columns = [c.strip() for c in raw.columns]
    raw_idx = np.concatenate([np.load(os.path.join(DATA, ds, f'idx_{s}.npy'))
                              for s in SPLITS])
    assert len(raw_idx) == len(y) == len(raw), 'raw file size mismatch'

    # Verify the raw file is the same one the pipeline was processed from
    # before trusting its timedelta column.
    for j, col in enumerate(info_src['num_col_names']):
        assert np.allclose(X_num[:, j], raw[col].to_numpy()[raw_idx].astype(np.float32),
                           rtol=1e-5), f'raw alignment failed on {col}'
    assert np.allclose(y, raw['shares'].to_numpy()[raw_idx].astype(np.float32)), \
        'raw alignment failed on shares'

    timedelta = raw['timedelta'].to_numpy()[raw_idx].astype(np.float32)
    new_num = np.column_stack([timedelta, X_num]).astype(np.float32)  # 45 cols

    dummies = X_cat.astype(np.int64)
    data_channel = dummies[:, 0:6].argmax(axis=1)    # TabDiff cat_col1
    weekday = dummies[:, 6:14].argmax(axis=1)        # TabDiff cat_col2
    new_cat = np.column_stack([data_channel, weekday]).astype(np.str_)

    for split in SPLITS:
        rows = np.load(os.path.join(dst, f'idx_{split}.npy'))
        old_y = np.load(os.path.join(dst, f'y_{split}.npy'), allow_pickle=True)
        assert np.array_equal(old_y, y[rows]), f'news {split}: row assignment drifted'
        _write_split(dst, split, new_num, new_cat, y, rows)

    _update_info(dst,
                 ['timedelta'] + info_src['num_col_names'],
                 ['data_channel', 'weekday'],
                 'Feature schema matches TabDiff preprocess_news: timedelta kept '
                 '(re-attached from raw OnlineNewsPopularity.csv, alignment '
                 'verified), 14 one-hot dummies collapsed to data_channel/weekday '
                 'via argmax. Rows unchanged.')
    print('  [news] 45 num / 2 cat written (was 44 num / 14 cat); rows unchanged')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--news_raw', type=str,
                   default=os.path.join(DATA, 'raw', 'OnlineNewsPopularity.csv'),
                   help='Path to the raw OnlineNewsPopularity.csv (needed to '
                        're-attach timedelta for news; see the docstring for '
                        'the committed copy\'s provenance)')
    args = p.parse_args()

    apply_beijing()
    if os.path.exists(args.news_raw):
        apply_news(args.news_raw)
    else:
        print(f'  [news] SKIPPED — raw CSV not found at {args.news_raw}')
    print('Done. Original data/beijing/ and data/news/ untouched.')
