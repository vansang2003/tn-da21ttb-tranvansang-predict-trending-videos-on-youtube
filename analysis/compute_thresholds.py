import os
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score

DATA_PATH = os.path.join('data', 'data_youtube_trending_video.csv')
REPORTS_DIR = os.path.join('reports', 'thresholds')
JSON_OUT = os.path.join(REPORTS_DIR, 'thresholds_config.json')


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def text_length(s):
        return len(str(s))
    def word_count(s):
        return len(str(s).split())
    def tag_count(s):
        return len(str(s).split(','))

    for col in ['viewCount', 'likeCount', 'commentCount']:
        df[col] = pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0)

    df['title_len'] = df['title'].apply(text_length)
    df['desc_len'] = df['description'].apply(text_length)
    df['desc_words'] = df['description'].apply(word_count)
    df['tag_count'] = df['tags'].apply(tag_count)

    # Rates
    df['like_rate'] = np.where(df['viewCount'] > 0, df['likeCount'] / df['viewCount'] * 100.0, 0.0)
    df['comment_rate'] = np.where(df['viewCount'] > 0, df['commentCount'] / df['viewCount'] * 100.0, 0.0)
    return df


def segment_video_type(df: pd.DataFrame) -> pd.Series:
    # Nếu có cột videoType/shorts thì dùng, nếu không tạm coi tất cả là "regular"
    for cand in ['videoType', 'type', 'isShort']:
        if cand in df.columns:
            if cand == 'isShort':
                return df[cand].astype(int).map({1: 'short', 0: 'regular'})
            return df[cand].astype(str)
    return pd.Series(['regular'] * len(df), index=df.index)


def optimal_threshold_by_youden(y_true: np.ndarray, x_values: np.ndarray, greater_is_positive: bool = True) -> Dict:
    if not greater_is_positive:
        x_values = -x_values
    fpr, tpr, thr = roc_curve(y_true, x_values)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    best_thr = thr[idx]
    if not greater_is_positive:
        best_thr = -best_thr
    roc_auc = auc(fpr, tpr)
    return {
        'threshold': float(best_thr),
        'youden_j': float(youden[idx]),
        'tpr': float(tpr[idx]),
        'fpr': float(fpr[idx]),
        'roc_auc': float(roc_auc)
    }


def bootstrap_ci(values: np.ndarray, alpha: float = 0.05, n_boot: int = 1000) -> Tuple[float, float]:
    if len(values) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(np.mean(sample))
    lower = np.percentile(boots, 100 * (alpha / 2))
    upper = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def compute_woe_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 10) -> Dict:
    df = df[[feature, target]].dropna().copy()
    if df[feature].nunique() <= 1:
        return {'iv': 0.0, 'bins': []}
    df['bin'] = pd.qcut(df[feature], q=min(bins, df[feature].nunique()), duplicates='drop')
    grp = df.groupby('bin')[target].agg(['count', 'sum'])
    grp['non_event'] = grp['count'] - grp['sum']
    total_event = grp['sum'].sum()
    total_non_event = grp['non_event'].sum()
    # Laplace smoothing
    grp['rate_event'] = (grp['sum'] + 0.5) / (total_event + 1.0)
    grp['rate_non_event'] = (grp['non_event'] + 0.5) / (total_non_event + 1.0)
    grp['woe'] = np.log(grp['rate_event'] / grp['rate_non_event'])
    grp['iv_comp'] = (grp['rate_event'] - grp['rate_non_event']) * grp['woe']
    iv = float(grp['iv_comp'].sum())
    bins_out = []
    for idx, row in grp.reset_index().iterrows():
        bins_out.append({
            'bin': str(row['bin']),
            'count': int(row['count']),
            'event': int(row['sum']),
            'non_event': int(row['non_event']),
            'woe': float(row['woe'])
        })
    return {'iv': iv, 'bins': bins_out}


def analyze_segment(df: pd.DataFrame, target: str, segment_name: str) -> Dict:
    print(f'Phân tích segment: {segment_name}')
    results = {'segment': segment_name, 'features': {}}
    candidates = [
        ('viewCount', True),
        ('likeCount', True),
        ('commentCount', True),
        ('like_rate', True),
        ('comment_rate', True),
        ('tag_count', True),
        ('title_len', True),
        ('desc_words', True)
    ]
    for feat, greater_pos in candidates:
        if feat not in df.columns:
            continue
        x = pd.to_numeric(df[feat], errors='coerce').fillna(0).values
        y = pd.to_numeric(df[target], errors='coerce').fillna(0).astype(int).values
        # Ngưỡng tối ưu đơn biến theo Youden
        youden = optimal_threshold_by_youden(y, x, greater_is_positive=greater_pos)
        # CI thô bằng bootstrap trên giá trị trong/ngoài ngưỡng (trung bình y)
        thr = youden['threshold']
        if greater_pos:
            in_mask = x >= thr
        else:
            in_mask = x <= thr
        pos_rate = y[in_mask]
        neg_rate = y[~in_mask]
        ci_in = bootstrap_ci(pos_rate)
        ci_out = bootstrap_ci(neg_rate)
        # WOE/IV theo quantile bins
        woe_iv = compute_woe_iv(df, feat, target)
        results['features'][feat] = {
            'threshold_youden': youden,
            'pos_rate_ci_in': ci_in,
            'pos_rate_ci_out': ci_out,
            'iv': woe_iv['iv'],
            'woe_bins': woe_iv['bins']
        }
    return results


def main():
    ensure_dirs()
    print('Đọc dữ liệu...')
    df = pd.read_csv(DATA_PATH)
    if 'isTrending' not in df.columns:
        raise RuntimeError('Thiếu cột isTrending trong dữ liệu')
    print('Sinh đặc trưng...')
    df = engineer_features(df)
    print('Tách loại video...')
    df['video_segment'] = segment_video_type(df)

    output = {'segments': []}
    for seg_name, seg_df in df.groupby('video_segment'):
        res = analyze_segment(seg_df, 'isTrending', seg_name)
        output['segments'].append(res)

    print('Lưu JSON cấu hình ngưỡng...')
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f'Đã lưu {JSON_OUT}')


if __name__ == '__main__':
    main()


