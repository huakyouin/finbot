import pandas as pd
import numpy as np
import re
import abc
from qlib.data._libs.rolling import rolling_slope, rolling_rsquare, rolling_resi
from qlib.data._libs.expanding import expanding_slope, expanding_rsquare, expanding_resi

class Expression(abc.ABC):
    
    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return str(self)

    def __gt__(self, other):
        return Gt(self, other)

    def __ge__(self, other):
        return Ge(self, other)

    def __lt__(self, other):
        return Lt(self, other)

    def __le__(self, other):
        return Le(self, other)

    def __eq__(self, other):
        return Eq(self, other)

    def __ne__(self, other):
        return Ne(self, other)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(self, other)

    def __div__(self, other):
        return Div(self, other)

    def __rdiv__(self, other):
        return Div(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __pow__(self, other):
        return Power(self, other)

    def __rpow__(self, other):
        return Power(other, self)

    def __and__(self, other):
        return And(self, other)

    def __rand__(self, other):
        return And(other, self)

    def __or__(self, other):
        return Or(self, other)

    def __ror__(self, other):
        return Or(other, self)


    def load(self, instrument, start_index, end_index, *args):
        series = self._load_internal(instrument, start_index, end_index, *args)
        return series

    @abc.abstractmethod
    def _load_internal(self, instrument, start_index, end_index, *args) -> pd.Series:
        raise NotImplementedError("This function must be implemented in your newly defined feature")

class Feature(Expression):
    """Static Expression

    This kind of feature will load data from provider
    """

    def __init__(self, name=None):
        if name:
            self._name = name
        else:
            self._name = type(self).__name__

    def __str__(self):
        return "$" + self._name

    def _load_internal(self, instrument, start_index, end_index):
        return instrument.loc[start_index:end_index][self._name]


class ExpressionOps(Expression):
    pass

class ElemOperator(ExpressionOps):
    """Element-wise Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, feature):
        self.feature = feature

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.feature)

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return self.feature.get_extended_window_size()


class ChangeInstrument(ElemOperator):
    """Change Instrument Operator
    In some case, one may want to change to another instrument when calculating, for example, to
    calculate beta of a stock with respect to a market index.
    This would require changing the calculation of features from the stock (original instrument) to
    the index (reference instrument)
    Parameters
    ----------
    instrument: new instrument for which the downstream operations should be performed upon.
                i.e., SH000300 (CSI300 index), or ^GPSC (SP500 index).

    feature: the feature to be calculated for the new instrument.
    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, instrument, feature):
        self.instrument = instrument
        self.feature = feature

    def __str__(self):
        return "{}('{}',{})".format(type(self).__name__, self.instrument, self.feature)

    def load(self, instrument, start_index, end_index, *args):
        # the first `instrument` is ignored
        return super().load(self.instrument, start_index, end_index, *args)

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.feature.load(instrument, start_index, end_index, *args)


class NpElemOperator(ElemOperator):
    """Numpy Element-wise Operator

    Parameters
    ----------
    feature : Expression
        feature instance
    func : str
        numpy feature operation method

    Returns
    ----------
    Expression
        feature operation output
    """

    def __init__(self, feature, func):
        self.func = func
        super(NpElemOperator, self).__init__(feature)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        return getattr(np, self.func)(series)


class Abs(NpElemOperator):
    """Feature Absolute Value

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with absolute output
    """

    def __init__(self, feature):
        super(Abs, self).__init__(feature, "abs")


class Sign(NpElemOperator):
    """Feature Sign

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with sign
    """

    def __init__(self, feature):
        super(Sign, self).__init__(feature, "sign")

    def _load_internal(self, instrument, start_index, end_index, *args):
        """
        To avoid error raised by bool type input, we transform the data into float32.
        """
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO:  More precision types should be configurable
        series = series.astype(np.float32)
        return getattr(np, self.func)(series)


class Log(NpElemOperator):
    """Feature Log

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Expression
        a feature instance with log
    """

    def __init__(self, feature):
        super(Log, self).__init__(feature, "log")


class Mask(NpElemOperator):
    """Feature Mask

    Parameters
    ----------
    feature : Expression
        feature instance
    instrument : str
        instrument mask

    Returns
    ----------
    Expression
        a feature instance with masked instrument
    """

    def __init__(self, feature, instrument):
        super(Mask, self).__init__(feature, "mask")
        self.instrument = instrument

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.instrument.lower())

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.feature.load(self.instrument, start_index, end_index, *args)


class Not(NpElemOperator):
    """Not Operator

    Parameters
    ----------
    feature : Expression
        feature instance

    Returns
    ----------
    Feature:
        feature elementwise not output
    """

    def __init__(self, feature):
        super(Not, self).__init__(feature, "bitwise_not")


#################### Pair-Wise Operator ####################
class PairOperator(ExpressionOps):
    """Pair-wise operator

    Parameters
    ----------
    feature_left : Expression
        feature instance or numeric value
    feature_right : Expression
        feature instance or numeric value

    Returns
    ----------
    Feature:
        two features' operation output
    """

    def __init__(self, feature_left, feature_right):
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature_left, self.feature_right)

    def get_longest_back_rolling(self):
        if isinstance(self.feature_left, (Expression,)):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, (Expression,)):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0
        return max(left_br, right_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, (Expression,)):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0

        if isinstance(self.feature_right, (Expression,)):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        return max(ll, rl), max(lr, rr)


class NpPairOperator(PairOperator):
    def __init__(self, feature_left, feature_right, func):
        self.func = func
        super(NpPairOperator, self).__init__(feature_left, feature_right)

    def _load_internal(self, instrument, start_index, end_index, *args):
        if isinstance(self.feature_left, (Expression,)):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, (Expression,)):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        res = getattr(np, self.func)(series_left, series_right)
        return res


class Power(NpPairOperator):
    """Power Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        The bases in feature_left raised to the exponents in feature_right
    """

    def __init__(self, feature_left, feature_right):
        super(Power, self).__init__(feature_left, feature_right, "power")


class Add(NpPairOperator):
    """Add Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' sum
    """

    def __init__(self, feature_left, feature_right):
        super(Add, self).__init__(feature_left, feature_right, "add")


class Sub(NpPairOperator):
    """Subtract Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' subtraction
    """

    def __init__(self, feature_left, feature_right):
        super(Sub, self).__init__(feature_left, feature_right, "subtract")


class Mul(NpPairOperator):
    """Multiply Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' product
    """

    def __init__(self, feature_left, feature_right):
        super(Mul, self).__init__(feature_left, feature_right, "multiply")


class Div(NpPairOperator):
    """Division Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' division
    """

    def __init__(self, feature_left, feature_right):
        super(Div, self).__init__(feature_left, feature_right, "divide")


class Greater(NpPairOperator):
    """Greater Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        greater elements taken from the input two features
    """

    def __init__(self, feature_left, feature_right):
        super(Greater, self).__init__(feature_left, feature_right, "maximum")


class Less(NpPairOperator):
    """Less Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        smaller elements taken from the input two features
    """

    def __init__(self, feature_left, feature_right):
        super(Less, self).__init__(feature_left, feature_right, "minimum")


class Gt(NpPairOperator):
    """Greater Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left > right`
    """

    def __init__(self, feature_left, feature_right):
        super(Gt, self).__init__(feature_left, feature_right, "greater")


class Ge(NpPairOperator):
    """Greater Equal Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left >= right`
    """

    def __init__(self, feature_left, feature_right):
        super(Ge, self).__init__(feature_left, feature_right, "greater_equal")


class Lt(NpPairOperator):
    """Less Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left < right`
    """

    def __init__(self, feature_left, feature_right):
        super(Lt, self).__init__(feature_left, feature_right, "less")


class Le(NpPairOperator):
    """Less Equal Than Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left <= right`
    """

    def __init__(self, feature_left, feature_right):
        super(Le, self).__init__(feature_left, feature_right, "less_equal")


class Eq(NpPairOperator):
    """Equal Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left == right`
    """

    def __init__(self, feature_left, feature_right):
        super(Eq, self).__init__(feature_left, feature_right, "equal")


class Ne(NpPairOperator):
    """Not Equal Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        bool series indicate `left != right`
    """

    def __init__(self, feature_left, feature_right):
        super(Ne, self).__init__(feature_left, feature_right, "not_equal")


class And(NpPairOperator):
    """And Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' row by row & output
    """

    def __init__(self, feature_left, feature_right):
        super(And, self).__init__(feature_left, feature_right, "bitwise_and")


class Or(NpPairOperator):
    """Or Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance

    Returns
    ----------
    Feature:
        two features' row by row | outputs
    """

    def __init__(self, feature_left, feature_right):
        super(Or, self).__init__(feature_left, feature_right, "bitwise_or")


#################### Triple-wise Operator ####################
class If(ExpressionOps):
    """If Operator

    Parameters
    ----------
    condition : Expression
        feature instance with bool values as condition
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    """

    def __init__(self, condition, feature_left, feature_right):
        self.condition = condition
        self.feature_left = feature_left
        self.feature_right = feature_right

    def __str__(self):
        return "If({},{},{})".format(self.condition, self.feature_left, self.feature_right)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series_cond = self.condition.load(instrument, start_index, end_index, *args)
        if isinstance(self.feature_left, (Expression,)):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left
        if isinstance(self.feature_right, (Expression,)):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right
        series = pd.Series(np.where(series_cond, series_left, series_right), index=series_cond.index)
        return series

    def get_longest_back_rolling(self):
        if isinstance(self.feature_left, (Expression,)):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, (Expression,)):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0

        if isinstance(self.condition, (Expression,)):
            c_br = self.condition.get_longest_back_rolling()
        else:
            c_br = 0
        return max(left_br, right_br, c_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, (Expression,)):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0

        if isinstance(self.feature_right, (Expression,)):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0

        if isinstance(self.condition, (Expression,)):
            cl, cr = self.condition.get_extended_window_size()
        else:
            cl, cr = 0, 0
        return max(ll, rl, cl), max(lr, rr, cr)


#################### Rolling ####################
# NOTE: methods like `rolling.mean` are optimized with cython,
# and are super faster than `rolling.apply(np.mean)`


class Rolling(ExpressionOps):
    """Rolling Operator
    The meaning of rolling and expanding is the same in pandas.
    When the window is set to 0, the behaviour of the operator should follow `expanding`
    Otherwise, it follows `rolling`

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size
    func : str
        rolling method

    Returns
    ----------
    Expression
        rolling outputs
    """

    def __init__(self, feature, N, func):
        self.feature = feature
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # NOTE: remove all null check,
        # now it's user's responsibility to decide whether use features in null days
        # isnull = series.isnull() # NOTE: isnull = NaN, inf is not null
        if isinstance(self.N, int) and self.N == 0:
            series = getattr(series.expanding(min_periods=1), self.func)()
        elif isinstance(self.N, float) and 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = getattr(series.rolling(self.N, min_periods=1), self.func)()
            # series.iloc[:self.N-1] = np.nan
        # series[isnull] = np.nan
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if 0 < self.N < 1:
            return int(np.log(1e-6) / np.log(1 - self.N))  # (1 - N)**window == 1e-6
        return self.feature.get_longest_back_rolling() + self.N - 1

    def get_extended_window_size(self):
        if self.N == 0:
            # FIXME: How to make this accurate and efficiently? Or  should we
            # remove such support for N == 0?
            print("The Rolling(ATTR, 0) will not be accurately calculated")
            return self.feature.get_extended_window_size()
        elif 0 < self.N < 1:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            size = int(np.log(1e-6) / np.log(1 - self.N))
            lft_etd = max(lft_etd + size - 1, lft_etd)
            return lft_etd, rght_etd
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N - 1, lft_etd)
            return lft_etd, rght_etd


class Ref(Rolling):
    """Feature Reference

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        N = 0, retrieve the first data; N > 0, retrieve data of N periods ago; N < 0, future data

    Returns
    ----------
    Expression
        a feature instance with target reference
    """

    def __init__(self, feature, N):
        super(Ref, self).__init__(feature, N, "ref")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # N = 0, return first day
        if series.empty:
            return series  # Pandas bug, see: https://github.com/pandas-dev/pandas/issues/21049
        elif self.N == 0:
            series = pd.Series(series.iloc[0], index=series.index)
        else:
            series = series.shift(self.N)  # copy
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        return self.feature.get_longest_back_rolling() + self.N

    def get_extended_window_size(self):
        if self.N == 0:
            print("The Ref(ATTR, 0) will not be accurately calculated")
            return self.feature.get_extended_window_size()
        else:
            lft_etd, rght_etd = self.feature.get_extended_window_size()
            lft_etd = max(lft_etd + self.N, lft_etd)
            rght_etd = max(rght_etd - self.N, rght_etd)
            return lft_etd, rght_etd


class Mean(Rolling):
    """Rolling Mean (MA)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling average
    """

    def __init__(self, feature, N):
        super(Mean, self).__init__(feature, N, "mean")


class Sum(Rolling):
    """Rolling Sum

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling sum
    """

    def __init__(self, feature, N):
        super(Sum, self).__init__(feature, N, "sum")


class Std(Rolling):
    """Rolling Std

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling std
    """

    def __init__(self, feature, N):
        super(Std, self).__init__(feature, N, "std")


class Var(Rolling):
    """Rolling Variance

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling variance
    """

    def __init__(self, feature, N):
        super(Var, self).__init__(feature, N, "var")


class Skew(Rolling):
    """Rolling Skewness

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling skewness
    """

    def __init__(self, feature, N):
        if N != 0 and N < 3:
            raise ValueError("The rolling window size of Skewness operation should >= 3")
        super(Skew, self).__init__(feature, N, "skew")


class Kurt(Rolling):
    """Rolling Kurtosis

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling kurtosis
    """

    def __init__(self, feature, N):
        if N != 0 and N < 4:
            raise ValueError("The rolling window size of Kurtosis operation should >= 5")
        super(Kurt, self).__init__(feature, N, "kurt")


class Max(Rolling):
    """Rolling Max

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max
    """

    def __init__(self, feature, N):
        super(Max, self).__init__(feature, N, "max")


class IdxMax(Rolling):
    """Rolling Max Index

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max index
    """

    def __init__(self, feature, N):
        super(IdxMax, self).__init__(feature, N, "idxmax")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmax() + 1, raw=True)
        return series


class Min(Rolling):
    """Rolling Min

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling min
    """

    def __init__(self, feature, N):
        super(Min, self).__init__(feature, N, "min")


class IdxMin(Rolling):
    """Rolling Min Index

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling min index
    """

    def __init__(self, feature, N):
        super(IdxMin, self).__init__(feature, N, "idxmin")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(lambda x: x.argmin() + 1, raw=True)
        return series


class Quantile(Rolling):
    """Rolling Quantile

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling quantile
    """

    def __init__(self, feature, N, qscore):
        super(Quantile, self).__init__(feature, N, "quantile")
        self.qscore = qscore

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature, self.N, self.qscore)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series.expanding(min_periods=1).quantile(self.qscore)
        else:
            series = series.rolling(self.N, min_periods=1).quantile(self.qscore)
        return series


class Med(Rolling):
    """Rolling Median

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling median
    """

    def __init__(self, feature, N):
        super(Med, self).__init__(feature, N, "median")


class Mad(Rolling):
    """Rolling Mean Absolute Deviation

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling mean absolute deviation
    """

    def __init__(self, feature, N):
        super(Mad, self).__init__(feature, N, "mad")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO: implement in Cython

        def mad(x):
            x1 = x[~np.isnan(x)]
            return np.mean(np.abs(x1 - x1.mean()))

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(mad, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(mad, raw=True)
        return series


class Rank(Rolling):
    """Rolling Rank (Percentile)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling rank
    """

    def __init__(self, feature, N):
        super(Rank, self).__init__(feature, N, "rank")

    # for compatiblity of python 3.7, which doesn't support pandas 1.4.0+ which implements Rolling.rank
    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        rolling_or_expending = series.expanding(min_periods=1) if self.N == 0 else series.rolling(self.N, min_periods=1)
        if hasattr(rolling_or_expending, "rank"):
            return rolling_or_expending.rank(pct=True)

        def rank(x):
            if np.isnan(x[-1]):
                return np.nan
            x1 = x[~np.isnan(x)]
            if x1.shape[0] == 0:
                return np.nan
            return percentileofscore(x1, x1[-1]) / 100

        return rolling_or_expending.apply(rank, raw=True)


class Count(Rolling):
    """Rolling Count

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling count of number of non-NaN elements
    """

    def __init__(self, feature, N):
        super(Count, self).__init__(feature, N, "count")


class Delta(Rolling):
    """Rolling Delta

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with end minus start in rolling window
    """

    def __init__(self, feature, N):
        super(Delta, self).__init__(feature, N, "delta")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = series - series.iloc[0]
        else:
            series = series - series.shift(self.N)
        return series


class Slope(Rolling):
    """Rolling Slope

    Usage Example:
    - "Slope($close, %d)/$close"

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with linear regression slope of given window
    """

    def __init__(self, feature, N):
        super(Slope, self).__init__(feature, N, "slope")
    
    def _load_internal(self, instrument, start_index, end_index, *args):
        def calc_slope(x):
            x = x / x[0]  # 这里做了一个“归一化”
            slope = np.polyfit(range(len(x)), x, 1)[0]
            return slope
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = series.rolling(self.N, min_periods=2).apply(calc_slope)
        series = pd.Series(result, index=series.index)
        return series


class Rsquare(Rolling):
    """Rolling R-value Square

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with linear regression r-value square of given window
    """

    def __init__(self, feature, N):
        super(Rsquare, self).__init__(feature, N, "rsquare")

    def _load_internal(self, instrument, start_index, end_index, *args):
        _series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_rsquare(_series.values), index=_series.index)
        else:
            series = pd.Series(rolling_rsquare(_series.values, self.N), index=_series.index)
            series.loc[np.isclose(_series.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)] = np.nan
        return series
    

class Resi(Rolling):
    """Rolling Regression Residuals

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression residuals of given window
    """

    def __init__(self, feature, N):
        super(Resi, self).__init__(feature, N, "resi")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        if self.N == 0:
            series = pd.Series(expanding_resi(series.values), index=series.index)
        else:
            series = pd.Series(rolling_resi(series.values, self.N), index=series.index)
        return series


class WMA(Rolling):
    """Rolling WMA

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with weighted moving average output
    """

    def __init__(self, feature, N):
        super(WMA, self).__init__(feature, N, "wma")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        # TODO: implement in Cython

        def weighted_mean(x):
            w = np.arange(len(x)) + 1
            w = w / w.sum()
            return np.nanmean(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(weighted_mean, raw=True)
        else:
            series = series.rolling(self.N, min_periods=1).apply(weighted_mean, raw=True)
        return series


class EMA(Rolling):
    """Rolling Exponential Mean (EMA)

    Parameters
    ----------
    feature : Expression
        feature instance
    N : int, float
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with regression r-value square of given window
    """

    def __init__(self, feature, N):
        super(EMA, self).__init__(feature, N, "ema")

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        def exp_weighted_mean(x):
            a = 1 - 2 / (1 + len(x))
            w = a ** np.arange(len(x))[::-1]
            w /= w.sum()
            return np.nansum(w * x)

        if self.N == 0:
            series = series.expanding(min_periods=1).apply(exp_weighted_mean, raw=True)
        elif 0 < self.N < 1:
            series = series.ewm(alpha=self.N, min_periods=1).mean()
        else:
            series = series.ewm(span=self.N, min_periods=1).mean()
        return series


#################### Pair-Wise Rolling ####################
class PairRolling(ExpressionOps):
    """Pair Rolling Operator

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling output of two input features
    """

    def __init__(self, feature_left, feature_right, N, func):
        # TODO: in what case will a const be passed into `__init__` as `feature_left` or `feature_right`
        self.feature_left = feature_left
        self.feature_right = feature_right
        self.N = N
        self.func = func

    def __str__(self):
        return "{}({},{},{})".format(type(self).__name__, self.feature_left, self.feature_right, self.N)

    def _load_internal(self, instrument, start_index, end_index, *args):
        assert any(
            [isinstance(self.feature_left, Expression), self.feature_right, Expression]
        ), "at least one of two inputs is Expression instance"

        if isinstance(self.feature_left, Expression):
            series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        else:
            series_left = self.feature_left  # numeric value
        if isinstance(self.feature_right, Expression):
            series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        else:
            series_right = self.feature_right

        if self.N == 0:
            series = getattr(series_left.expanding(min_periods=1), self.func)(series_right)
        else:
            series = getattr(series_left.rolling(self.N, min_periods=1), self.func)(series_right)
        return series

    def get_longest_back_rolling(self):
        if self.N == 0:
            return np.inf
        if isinstance(self.feature_left, Expression):
            left_br = self.feature_left.get_longest_back_rolling()
        else:
            left_br = 0

        if isinstance(self.feature_right, Expression):
            right_br = self.feature_right.get_longest_back_rolling()
        else:
            right_br = 0
        return max(left_br, right_br)

    def get_extended_window_size(self):
        if isinstance(self.feature_left, Expression):
            ll, lr = self.feature_left.get_extended_window_size()
        else:
            ll, lr = 0, 0
        if isinstance(self.feature_right, Expression):
            rl, rr = self.feature_right.get_extended_window_size()
        else:
            rl, rr = 0, 0
        if self.N == 0:
            print("The PairRolling(ATTR, 0) will not be accurately calculated")
            return -np.inf, max(lr, rr)
        else:
            return max(ll, rl) + self.N - 1, max(lr, rr)


class Corr(PairRolling):
    """Rolling Correlation

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling correlation of two input features
    """

    def __init__(self, feature_left, feature_right, N):
        super(Corr, self).__init__(feature_left, feature_right, N, "corr")

    def _load_internal(self, instrument, start_index, end_index, *args):
        res: pd.Series = super(Corr, self)._load_internal(instrument, start_index, end_index, *args)

        # NOTE: Load uses MemCache, so calling load again will not cause performance degradation
        series_left = self.feature_left.load(instrument, start_index, end_index, *args)
        series_right = self.feature_right.load(instrument, start_index, end_index, *args)
        res.loc[
            np.isclose(series_left.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
            | np.isclose(series_right.rolling(self.N, min_periods=1).std(), 0, atol=2e-05)
        ] = np.nan
        return res


class Cov(PairRolling):
    """Rolling Covariance

    Parameters
    ----------
    feature_left : Expression
        feature instance
    feature_right : Expression
        feature instance
    N : int
        rolling window size

    Returns
    ----------
    Expression
        a feature instance with rolling max of two input features
    """

    def __init__(self, feature_left, feature_right, N):
        super(Cov, self).__init__(feature_left, feature_right, N, "cov")


#################### Operator which only support data with time index ####################
# Convention
# - The name of the operators in this section will start with "T"


class TResample(ElemOperator):
    def __init__(self, feature, freq, func):
        """
        Resampling the data to target frequency.
        The resample function of pandas is used.

        - the timestamp will be at the start of the time span after resample.

        Parameters
        ----------
        feature : Expression
            An expression for calculating the feature
        freq : str
            It will be passed into the resample method for resampling basedn on given frequency
        func : method
            The method to get the resampled values
            Some expression are high frequently used
        """
        self.feature = feature
        self.freq = freq
        self.func = func

    def __str__(self):
        return "{}({},{})".format(type(self).__name__, self.feature, self.freq)

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        if series.empty:
            return series
        else:
            if self.func == "sum":
                return getattr(series.resample(self.freq), self.func)(min_count=1)
            else:
                return getattr(series.resample(self.freq), self.func)()