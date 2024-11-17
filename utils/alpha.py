import polars as pl
from polars_ta.prefix.tdx import *
from polars_ta.prefix.wq import *
import numpy as np

#############################
## 特征转换
#############################
OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, VWAP = [pl.col(col) for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap']]

def fast_linregress(x, y):
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            slope = np.dot(x - x_mean, y - y_mean) / np.dot(x - x_mean, x - x_mean)
            intercept = y_mean - slope * x_mean
            y_pred = slope * x + intercept
            ss_total = np.sum((y - np.mean(y)) ** 2) + 1e-12
            ss_residual = np.sum((y - y_pred) ** 2)
            r2 = 1 - (ss_residual / ss_total)
            resd = np.sum(y - y_pred)
            return slope, intercept, r2, resd

def build_alpha158(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(by=['datetime'])
    df = df.with_columns([
        ((CLOSE - OPEN) / OPEN).alias('KMID'),
        ((HIGH - LOW) / OPEN).alias("KLEN"),
        ((CLOSE - OPEN) / (HIGH - LOW + 1e-12)).alias("KMID2"),
        ((HIGH - max_(OPEN, CLOSE)) / OPEN).alias("KUP"),
        ((HIGH - max_(OPEN, CLOSE)) / (HIGH - LOW + 1e-12)).alias("KUP2"),
        ((min_(OPEN, CLOSE) - LOW) / OPEN).alias("KLOW"),
        ((min_(OPEN, CLOSE) - LOW) / (HIGH - LOW + 1e-12)).alias("KLOW2"),
        ((2 * CLOSE - HIGH - LOW) / OPEN).alias("KSFT"),
        ((2 * CLOSE - HIGH - LOW) / (HIGH - LOW + 1e-12)).alias("KSFT2"),
        *[(ts_delay(OPEN, i) / CLOSE).alias(f'OPEN{i}') for i in [0]],
        *[(ts_delay(HIGH, i) / CLOSE).alias(f'HIGH{i}') for i in [0]],
        *[(ts_delay(LOW, i) / CLOSE).alias(f'LOW{i}') for i in [0]],
        *[(ts_delay(VWAP, i) / CLOSE).alias(f'VWAP{i}') for i in [0]],
    ])
    for i in [5,10,20,30,60]:
        df = df.with_columns([
            (ts_delay(CLOSE, i) / CLOSE).alias(f'ROC{i}'),
            (ts_mean(CLOSE, i) / CLOSE).alias(f'MA{i}'),
            (CLOSE.rolling_std(i) / CLOSE).alias(f'STD{i}'),
            (CLOSE.rolling_max(i) / CLOSE).alias(f'MAX{i}'),
            (CLOSE.rolling_min(i) / CLOSE).alias(f'MIN{i}'),
            (CLOSE.rolling_quantile(0.8, interpolation='linear', window_size=i) / CLOSE).alias(f'QTLU{i}'),
            (CLOSE.rolling_quantile(0.2, interpolation='linear', window_size=i) / CLOSE).alias(f'QTLD{i}'),
            (ts_rank(CLOSE, i)).alias(f'RANK{i}'),
            (ts_RSV(HIGH, LOW, CLOSE, i)).alias(f'RSV{i}'),
            (1 - ts_arg_max(HIGH, i) / i).alias(f'IMAX{i}'),
            (1 - ts_arg_min(LOW, i) / i).alias(f'IMIN{i}'),
            (ts_corr(CLOSE, log1p(VOLUME), i)).alias(f'CORR{i}'),
            (ts_corr(CLOSE / ts_delay(CLOSE, 1), log1p(VOLUME / ts_delay(VOLUME, 1)), i)).alias(f'CORD{i}'),
            (ts_mean(CLOSE > ts_delay(CLOSE, 1), i)).alias(f'CNTP{i}'),
            (ts_mean(CLOSE < ts_delay(CLOSE, 1), i)).alias(f'CNTN{i}'),
            (ts_sum(max_(CLOSE - ts_delay(CLOSE, 1), 0), i) / (ts_sum(abs_(CLOSE - ts_delay(CLOSE, 1)), i) + 1e-12)).alias(f'SUMP{i}'),
            (ts_sum(max_(ts_delay(CLOSE, 1) - CLOSE, 0), i) / (ts_sum(abs_(CLOSE - ts_delay(CLOSE, 1)), i) + 1e-12)).alias(f'SUMN{i}'),
            (ts_mean(VOLUME, i) / (VOLUME + 1e-12)).alias(f'VMA{i}'),
            (VOLUME.rolling_std(i) / (VOLUME + 1e-12)).alias(f'VSTD{i}'),
            ((abs_(ts_returns(CLOSE, 1)) * VOLUME).rolling_std(i) / (ts_mean(abs_(ts_returns(CLOSE, 1)) * VOLUME, i) + 1e-12)).alias(f'WVMA{i}'),
            (ts_sum(max_(VOLUME - ts_delay(VOLUME, 1), 0), i) / (ts_sum(abs_(VOLUME - ts_delay(VOLUME, 1)), i) + 1e-12)).alias(f'VSUMP{i}'),
            (ts_sum(max_(ts_delay(VOLUME, 1) - VOLUME, 0), i) / (ts_sum(abs_(VOLUME - ts_delay(VOLUME, 1)), i) + 1e-12)).alias(f'VSUMN{i}'),
        ])
        df = df.with_columns([
            (pl.col(f"IMAX{i}") -pl.col(f"IMIN{i}")).alias(f"IMXD{i}"),
            (pl.col(f"CNTP{i}") - pl.col(f"CNTN{i}")).alias(f'CNTD{i}'),
            (pl.col(f"SUMP{i}") - pl.col(f"SUMN{i}")).alias(f'SUMD{i}'),
            (pl.col(f"VSUMP{i}") - pl.col(f"VSUMN{i}")).alias(f'VSUMD{i}'),
        ])

        reg = [fast_linregress(x = np.arange(i), y = df["close"][idx: idx + i].to_numpy()) for idx in range(len(df) - i + 1)]
        beta = [None] * (i - 1) + [item[0] for item in reg if item]
        rsqr = [None] * (i - 1) + [item[2] for item in reg if item]
        resi = [None] * (i - 1) + [item[3] for item in reg if item]
        row_n = len(df)
        df = df.with_columns([
            pl.Series(f'BETA{i}', beta[:row_n]),
            pl.Series(f'RSQR{i}', rsqr[:row_n]),
            pl.Series(f'RESI{i}', resi[:row_n]),
        ])
    
    return df

def build_label(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(by=['datetime'])
    df = df.with_columns([
        (CLOSE.shift(-1) / CLOSE - 1).alias("LABEL0")
    ])
    return df