"""Convert the raw datasets in ~/Desktop/new_datasets into the data/ layout
used by the rest of the pipeline (X_num/X_cat/y npy files + info.json),
with the same 64/16/20 train/val/test split scheme as the existing datasets.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = os.path.expanduser('~/Desktop/new_datasets')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
SEED = 0


def save_dataset(name, display_name, df, num_cols, cat_cols, y, task_type):
    out_dir = os.path.join(OUT, name)
    os.makedirs(out_dir, exist_ok=True)

    X_num = df[num_cols].to_numpy(dtype=np.float32)
    X_cat = df[cat_cols].astype(str).to_numpy(dtype=np.str_) if cat_cols else None
    y = np.asarray(y, dtype=np.float32 if task_type == 'regression' else np.int64)

    stratify = y if task_type != 'regression' else None
    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=0.2, random_state=SEED, stratify=stratify
    )
    stratify_tv = y[idx_trainval] if task_type != 'regression' else None
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=0.2, random_state=SEED, stratify=stratify_tv
    )

    for split, ids in [('train', idx_train), ('val', idx_val), ('test', idx_test)]:
        np.save(os.path.join(out_dir, f'X_num_{split}.npy'), X_num[ids])
        if X_cat is not None:
            np.save(os.path.join(out_dir, f'X_cat_{split}.npy'), X_cat[ids])
        np.save(os.path.join(out_dir, f'y_{split}.npy'), y[ids])
        np.save(os.path.join(out_dir, f'idx_{split}.npy'), ids)

    info = {
        'name': display_name,
        'id': f'{name}--default',
        'task_type': task_type,
        'n_num_features': len(num_cols),
        'n_cat_features': len(cat_cols),
        'num_col_names': num_cols,
        'cat_col_names': cat_cols,
        'train_size': len(idx_train),
        'val_size': len(idx_val),
        'test_size': len(idx_test),
    }
    with open(os.path.join(out_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)
    print(f'{name}: {len(idx_train)}/{len(idx_val)}/{len(idx_test)} '
          f'num={len(num_cols)} cat={len(cat_cols)} task={task_type}')


def process_magic():
    cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',
            'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
    df = pd.read_csv(os.path.join(RAW, 'magic+gamma+telescope', 'magic04.data'),
                     header=None, names=cols)
    y = (df['class'] == 'g').astype(int)
    save_dataset('magic', 'Magic Gamma Telescope', df, cols[:-1], [], y, 'binclass')


def process_shoppers():
    df = pd.read_csv(os.path.join(RAW, 'online_shoppers_intention.csv'))
    num_cols = ['Administrative', 'Administrative_Duration', 'Informational',
                'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
    cat_cols = ['Month', 'OperatingSystems', 'Browser', 'Region',
                'TrafficType', 'VisitorType', 'Weekend']
    y = df['Revenue'].astype(int)
    save_dataset('shoppers', 'Online Shoppers Intention', df, num_cols, cat_cols, y, 'binclass')


def process_beijing():
    df = pd.read_csv(os.path.join(RAW, 'PRSA_data_2010.1.1-2014.12.31.csv'))
    df = df.dropna(subset=['pm2.5']).reset_index(drop=True)
    num_cols = ['year', 'month', 'day', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    cat_cols = ['cbwd']
    y = df['pm2.5']
    save_dataset('beijing', 'Beijing PM2.5', df, num_cols, cat_cols, y, 'regression')


def process_news():
    df = pd.read_csv(os.path.join(RAW, 'OnlineNewsPopularity', 'OnlineNewsPopularity.csv'))
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=['url', 'timedelta'])
    cat_cols = [c for c in df.columns
                if c.startswith('data_channel_is_') or c.startswith('weekday_is_')
                or c == 'is_weekend']
    for c in cat_cols:
        df[c] = df[c].astype(int)
    num_cols = [c for c in df.columns if c not in cat_cols and c != 'shares']
    y = df['shares']
    save_dataset('news', 'Online News Popularity', df, num_cols, cat_cols, y, 'regression')


if __name__ == '__main__':
    process_magic()
    process_shoppers()
    process_beijing()
    process_news()
