
import numpy as np
import backtrader as bt

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
            # print(slope_period)
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