
import numpy as np
import polars as pl
from polars_ta.prefix.tdx import *
from polars_ta.prefix.wq import *
import backtrader as bt
import lightgbm as lgb

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

def func_ts_date(df: pl.DataFrame) -> pl.DataFrame:
    print(df['instrument'][0])
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
    df = df.with_columns([
        (CLOSE.shift(-1) / CLOSE - 1).alias("LABEL0")
    ])
    return df

#############################
## 模型
#############################
class BaseModel():
    _registry = {}

    @classmethod
    def register(cls, model_name):
        def decorator(subclass):
            cls._registry[model_name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def from_name(cls, model_name):
        subclass = cls._registry.get(model_name)
        assert subclass, f"No subclass registered for '{model_name}"
        return subclass()

    def train(self, df, feature_keys: list, label_key: str):
        raise NotImplementedError
    
    def pred(self, df, feature_keys):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


@BaseModel.register("lgb")
class LGBModel(BaseModel):
    def __init__(self):
        self.model_params = dict(
            objective="mse", 
            colsample_bytree=0.8879,
            learning_rate=0.0421,
            subsample=0.8789,
            lambda_l1=205.6999,
            lambda_l2=580.9768, # 正则超重
            max_depth=8,
            num_leaves=210,
            num_threads=20,
            verbosity=-1,
        )
        self.early_stopping_callback = lgb.early_stopping(50)
        self.verbose_eval_callback = lgb.log_evaluation(period=20)
        self.evals_result = {}
        self.evals_result_callback = lgb.record_evaluation(self.evals_result)

    def train(self, train_df, val_df, feature_keys, label_key):
        train_set = lgb.Dataset(train_df[feature_keys].values, label=train_df[label_key].values, free_raw_data=False)
        valid_set = lgb.Dataset(val_df[feature_keys].values, label=val_df[label_key].values, free_raw_data=False) 
        self.model = lgb.train(
            self.model_params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, valid_set],
            valid_names=['train', 'valid'],
            callbacks=[self.verbose_eval_callback, self.evals_result_callback],
        )

    def pred(self, df, feature_keys):
        df['predict'] = self.model.predict(df[feature_keys]).tolist()
        return df
    
    def load(self, path):
        self.model = lgb.Booster(model_file='path')

    def save(self, path):
        self.model.save_model('model.bin', format='binary')


#############################
## 回测
#############################
class MyPandasData(bt.feeds.PandasData):
    lines = ('predict',)
    params = (
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('predict', 'predict')
    )


class MyStrategy(bt.Strategy):
    '''
    根据过去20日的收盘价斜率来选择股票。
    如果持仓少于2个股票，且预测值为正，则买入斜率最高的60%的股票，并设置止盈止损。
    如果持仓超过2个股票，且预测值为负，则卖出股票。
    '''
    params=(
            ('stopup', 0.22), 
            ('stopdown', 0.08), 
            ('maperiod',15),
            ("lookback", 20),
            ('RSRS_period', 18),
            ('RSRS_avg_period', 600),
            )
    def __init__(self):
        # 初始化交易指令、买卖价格和手续费
        self.order = None
        self.buy_list = []

    def downcast(self, amount, lot):
        return abs(amount//lot*lot)
    

    #策略核心，根据条件执行买卖交易指令（必选）
    def next(self):
        # 记录收盘价
        if self.order: # 检查是否有指令等待执行, 
            return
        ### 计算股票池的动量（过去20日收盘价斜率）
        flag = False
        slope_period = []
        for code in set(self.getdatanames()):
            data = self.getdatabyname(code)
            closes_period = data.close.get(ago=-1, size=20).tolist()
            if len(closes_period) >= 20:
                flag = True
                slope = np.polyfit(range(len(closes_period)), closes_period, 1)[0]
            else:
                slope = float('inf')
            slope_period.append((code, slope))
        if flag:
            slope_period = sorted(slope_period, key=lambda x: x[1])
            slope_period = slope_period[::-1]
            print(slope_period)
            trade_codes = [x[0] for x in slope_period[:int(len(slope_period)*0.6)]]
        else:
            trade_codes = self.getdatanames()
        ###
        if len(self.buy_list) < 2:
            for code in set(trade_codes) - set(self.buy_list):
            # for code in set(self.getdatanames()) - set(self.buy_list):
                data = self.getdatabyname(code)
                price = data.close[0]
                price_up = price*(1 + self.params.stopup) # 止盈价
                price_down = price*(1-self.params.stopdown) # 止损价
                if data.predict[0] > 0:
                    order_value = self.broker.getvalue()*0.33
                    order_amount = self.downcast(order_value/data.close[0], 100)
                    self.order = self.buy_bracket(data, size=order_amount, name=code, limitprice = price_up, stopprice = price_down, exectype = bt.Order.Market)
                    # self.order = self.buy(data, size=order_amount, name=code)
                    self.buy_list.append(code)
        else:
        # elif self.position:
            now_list = []
            for code in self.buy_list:
                data = self.getdatabyname(code)
                if data.predict[0] < 0:
                    self.order = self.order_target_percent(data, 0, name=code)
                    continue
                now_list.append(code)
            self.buy_list = now_list.copy()