import abc

import numpy as np


def _metric_from_name(name, **kwargs):
    """
    Initialize metric from a name.

    Parameters
    ----------
    - name : str
        Name of the metric to initialize.

    Returns
    -------
    metric : Metric
    """
    dict_metric = {
        "return": Return,
        "average_return": AverageReturn,
        "final_wealth": FinalWealth,
        "drawdown": Drawdown,
        "max_drawdown": MaxDrawdown,
        "volatility": Volatility,
        "sharpe_ratio": SharpeRatio,
        "tradewise_sharpe_ratio": TradewiseSharpeRatio,
        "exposure": Exposure,
    }

    if name not in dict_metric.keys():
        raise ValueError

    return dict_metric[name](**kwargs)


class Metric(abc.ABC):
    """
    Base class of Metric.
    """

    @property
    @abc.abstractmethod
    def name(self):
        """
        Return name of self.
        """

    @abc.abstractmethod
    def result(self, strategy):
        """
        Evaluate metric of strategy.

        Parameters
        ----------
        - strategy : Strategy
        """

    def __call__(self, strategy, *args, **kwargs):
        return self.result(strategy, *args, **kwargs)


class Return(Metric):
    """
    Evaluate time-series of return.
    """

    def __init__(self, rate=False, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    @property
    def name(self):
        return "return"

    def _result_from_wealth(self, series_wealth):
        result = np.diff(series_wealth, prepend=series_wealth[0])

        if self.rate:
            result /= np.roll(series_wealth, 1)  # array_wealth[0] = 0.0

        return result

    def result(self, strategy, init_wealth=0.0):
        series_wealth = init_wealth + strategy.wealth().values
        return self._result_from_wealth(series_wealth)


class AverageReturn(Metric):
    """
    Evaluate time-series of return.

    Parameters
    ----------
    - rate : bool
    - n : int
    """

    def __init__(self, rate=False, n=1, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.n = n

    @property
    def name(self):
        return "average_return"

    def _result_from_wealth(self, series_wealth):
        n_bars = series_wealth.size

        if self.rate:
            total_return = series_wealth[-1] / series_wealth[0] - 1
            result = np.exp((self.n / (n_bars - 1)) * np.log(1 + total_return)) - 1.0
        else:
            total_return = series_wealth[-1] - series_wealth[0]
            result = (self.n / (n_bars - 1)) * total_return

        return result

    def result(self, strategy, init_wealth=0.0):
        series_wealth = init_wealth + strategy.wealth().values
        return self._result_from_wealth(series_wealth)


class FinalWealth(Metric):
    """
    Evaluate final wealth.

    Examples
    --------
    # >>> strategy.wealth
    # array([0.0, 1.0, 2.0, 1.0, 3.0])
    # >>> FinalWealth().result(strategy)
    # 3.0
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self):
        return "final_wealth"

    def _result_from_wealth(self, series_wealth):
        return series_wealth[-1]

    def result(self, strategy, init_wealth=0.0):
        series_wealth = init_wealth + strategy.wealth().values
        return self._result_from_wealth(series_wealth)


class Drawdown(Metric):
    """
    Evaluate series of drawdown rate/value of the wealth.

    Parameters
    ----------
    - rate : bool, default False
        If True, evaluate drawdown rate, that is, drawdown values
        divided by maximum values.
        If False, evaluate drawdown values.

    Returns
    -------
    drawdown : numpy.array, shape (n_bars, )

    Examples
    --------
    # >>> strategy.wealth
    # array([0.0, 1.0, 2.0, 1.0, 0.0])
    # >>> Drawdown().result(strategy)
    # array([0.0, 0.0, 0.0, -0.5, -1.0])
    # >>> Drawdown(rate=False).result(strategy)
    # array([0.0, 0.0, 0.0, -1.0, -2.0])
    """

    EPSILON = 10e-8

    def __init__(self, rate=False, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    @property
    def name(self):
        return "drawdown"

    def _result_from_wealth(self, series_wealth):
        cummax = np.maximum.accumulate(series_wealth)
        result = series_wealth - cummax

        if self.rate:
            result /= cummax + self.EPSILON

        return result

    def result(self, strategy, init_wealth=0.0):
        series_wealth = init_wealth + strategy.wealth().values
        return self._result_from_wealth(series_wealth)


class MaxDrawdown(Metric):
    """
    Evaluate maximum drawdown rate/value of the wealth.

    Parameters
    ----------
    - rate : bool, default False
        If True, evaluate maximum drawdown rate, that is, drawdown values
        divided by maximum values.
        If False, evaluate maximum drawdown values.

    Returns
    -------
    max_drawdown : float
        Maximum drop. always non-positive.
    """

    def __init__(self, rate=False, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    @property
    def name(self):
        return "max_drawdown"

    def result(self, strategy, init_wealth=0.0):
        return np.min(
            Drawdown(rate=self.rate).result(strategy, init_wealth=init_wealth)
        )


class Volatility(Metric):
    """
    Return standard deviation of pnl per bar.

    Parameters
    ----------
    - rate : bool, default False
        If True, volatility is evaluated for profit-loss rate.
        If False, volatility is evaluated for profit-loss.
    - n : int, default 1
        E.g. n = 365 (calendar days) for average annual return.

    Returns
    -------
    volatility : float
    """

    def __init__(self, rate=False, n=1, ddof=0, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.n = n
        self.ddof = ddof

    @property
    def name(self):
        return "volatility"

    def _result_from_wealth(self, series_wealth):
        series_return = Return(rate=self.rate)._result_from_wealth(series_wealth)
        result = np.sqrt(self.n) * np.std(series_return[1:], ddof=self.ddof)

        return result

    def result(self, strategy, init_wealth=0.0):
        series_wealth = init_wealth + strategy.wealth().values
        return self._result_from_wealth(series_wealth)


class SharpeRatio(Metric):
    """
    Evaluate sharpe ratio.

    Parameters
    ----------
    - rate : bool, default False
        If True, return and risk are evaluated using pnl rate.
        If False, return and risk are evaluated using pnl.
    - risk_free_return : float, default 0.0
        Risk free return.

    Returns
    -------
    sharpe_ratio : float
    """

    EPSILON = 10e-8

    def __init__(self, rate=False, n=1, risk_free_return=0.0, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self.n = n
        self.risk_free_return = risk_free_return

    @property
    def name(self):
        return "sharpe_ratio"

    def result(self, strategy, init_wealth=0.0):
        average_return = AverageReturn(rate=self.rate, n=self.n).result(
            strategy, init_wealth=init_wealth
        )
        volatility = Volatility(rate=self.rate, n=self.n).result(
            strategy, init_wealth=init_wealth
        )
        volatility = max(volatility, self.EPSILON)
        result = (average_return - self.risk_free_return) / volatility
        return result


class TradewiseSharpeRatio(Metric):
    """
    Evaluate Sharpe ratio for profit-loss of each trade.

    Returns
    -------
    tradewise_sharpe_ratio : float
    """

    EPSILON = 10e-8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self):
        return "tradewise_sharpe_ratio"

    def result(self, strategy):
        array_pnl = strategy.history.groupby("trade_id").agg(sum)["pnl"]
        avg_pnl = np.mean(array_pnl)
        std_pnl = np.std(array_pnl)  # TODO parameter ddof
        result = avg_pnl / max(std_pnl, self.EPSILON)

        return result


class Exposure(Metric):
    """
    Evaluate net exposure.

    Parameters
    ----------
    - net : bool, default False
        If True, evaluate net exposure.
        If False, evaluate absolute exposure.

    Returns
    -------
    exposure : numpy.array, shape (n_bars, )
    """

    def __init__(self, net=True, **kwargs):
        super().__init__(**kwargs)
        self.net = net

    @property
    def name(self):
        return "exposure"

    def result(self, strategy):
        exposure = np.zeros_like(strategy.universe.iloc[:, 0])

        for t in strategy.trades:
            i_open = strategy.universe.index.get_indexer([t.open_bar]).item()
            i_close = strategy.universe.index.get_indexer([t.close_bar]).item()

            value = t.array_value(strategy.universe).astype(exposure.dtype)
            value[:i_open] = 0
            value[i_close + 1 :] = 0

            if self.net:
                exposure += value.sum(axis=1)
            else:
                exposure += np.abs(value).sum(axis=1)

        return exposure


# class Beta(Metric):
#     """
#     Evaluate beta.

#     Parameters
#     ----------
#     - benchmark : strategy
#     - rate : bool
#     """

#     def __init__(self, benchmark, rate=False, **kwargs):
#         super().__init__(**kwargs)
#         self.rate = rate
#         self.benchmark = benchmark

#     @property
#     def name(self):
#         return "beta"

#     def result(self, strategy):
#         if not self.benchmark.is_run:
#             self.benchmark.run(strategy)

#         array_return_s = Return(rate=self.rate).result(strategy)
#         array_return_b = Return(rate=self.rate).result(benchmark)
#         cov = np.cov(array_return_s, array_return_b)[0, 1]
#         var = np.var(array_return_b)
#         result = cov / var

#         return result
