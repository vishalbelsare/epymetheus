import abc
from inspect import cleandoc
from time import time

import numpy as np
import pandas as pd

from .. import ts
from ..exceptions import NoTradeError
from ..exceptions import NotRunError
from ..metrics import metric_from_name


def create_strategy(logic_func, **params):
    """
    Initialize `Strategy` from function.

    Parameters
    ----------
    - logic_func : callable
        Function that returns iterable from universe and parameters.
    - **params
        Parameter values.

    Examples
    --------
    >>> from epymetheus import trade
    ...
    >>> def logic_func(universe, my_param):
    ...     return [my_param * trade("AAPL")]
    ...
    >>> strategy = create_strategy(logic_func, my_param=2.0)
    >>> universe = None
    >>> strategy(universe)
    [trade(['AAPL'], lot=[2.])]
    """
    return Strategy._create_strategy(logic_func=logic_func, params=params)


class Strategy(abc.ABC):
    """
    Base class of trading strategy.
    """

    def __init__(self, logic_func=None, params=None):
        if logic_func is not None:
            self.logic_func = logic_func
            self.params = params or {}

    @classmethod
    def _create_strategy(cls, logic_func, params):
        """
        Create strategy from a logic function.

        Parameters
        ----------
        - logic_func : callable
            Function that returns iterable from universe and parameters.
        - params : dict
            Parameter values.

        Returns
        -------
        strategy : Strategy
        """
        return cls(logic_func=logic_func, params=params)

    def __call__(self, universe, to_list=True):
        logic = self.get_logic()
        trades = logic(universe, **self.get_params())
        if to_list:
            trades = list(trades)
        return trades

    def logic(self, universe, **params):
        """
        Logic to generate trades from universe.
        Used to implement trading strategy by subclassing `Strategy`.

        Parameters
        ----------
        - universe : pandas.DataFrame
            Historical price data to apply this strategy.
            The index represents timestamps and the column is the assets.
        - **params
            Parameter values.

        Returns
        ------
        trades : iterable of trades
        """

    @property
    def name(self):
        """
        Return name of the strategy.
        """
        return self.__class__.__name__

    @property
    def description(self):
        """
        Return detailed description of the strategy.

        Returns
        -------
        description : str or None
            If strategy class has no docstring, return None.
        """
        if self.__class__.__doc__ is None:
            description = None
        else:
            description = cleandoc(self.__class__.__doc__)
        return description

    @property
    def n_trades(self):
        return len(self.trades)

    @property
    def n_orders(self):
        return sum(t.n_orders for t in self.trades)

    def history(self) -> pd.DataFrame:
        """
        Return `pandas.DataFrame` of trade history.

        Returns
        -------
        history : pandas.DataFrame
            Trade History.
        """
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        data = {}

        n_orders = np.array([t.n_orders for t in self.trades])

        data["trade_id"] = np.repeat(np.arange(len(self.trades)), n_orders)
        data["asset"] = np.concatenate([t.asset for t in self.trades])
        data["lot"] = np.concatenate([t.lot for t in self.trades])
        data["entry"] = np.repeat([t.entry for t in self.trades], n_orders)
        data["close"] = np.repeat([t.close for t in self.trades], n_orders)
        data["exit"] = np.repeat([t.exit for t in self.trades], n_orders)
        data["take"] = np.repeat([t.take for t in self.trades], n_orders)
        data["stop"] = np.repeat([t.stop for t in self.trades], n_orders)
        data["pnl"] = np.concatenate([t.final_pnl(self.universe) for t in self.trades])

        return pd.DataFrame(data)

    def wealth(self) -> pd.Series:
        """
        Return `pandas.Series` of wealth.

        Returns
        -------
        wealth : pandas.Series
            Series of wealth.
        """
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        return pd.Series(
            ts.wealth(self.trades, self.universe), index=self.universe.index
        )

    def drawdown(self) -> pd.Series:
        """
        Returns
        -------
        drawdown : pandas.Series
        """
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        drawdown = ts.drawdown(self.trades, self.universe)

        return pd.Series(drawdown, index=self.universe.index)

    def net_exposure(self) -> pd.Series:
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        exposure = ts.net_exposure(self.trades, self.universe)

        return pd.Series(exposure, index=self.universe.index)

    def abs_exposure(self) -> pd.Series:
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        exposure = ts.abs_exposure(self.trades, self.universe)

        return pd.Series(exposure, index=self.universe.index)

    def run(self, universe, verbose=True):
        """
        Run a backtesting of strategy.

        Parameters
        ----------
        - universe : pandas.DataFrame
            Historical price data to apply this strategy.
            The index represents timestamps and the column is the assets.
        - verbose : bool, default True
            Verbose mode.

        Returns
        -------
        self
        """
        _begin_time = time()

        self.universe = universe

        # Yield trades
        _begin_time_yield = time()
        trades = []
        for i, t in enumerate(self(universe, to_list=False) or []):
            if verbose:
                print(f"\r{i + 1} trades returned: {t} ... ", end="")
            trades.append(t)
        if len(trades) == 0:
            raise NoTradeError("No trade.")
        if verbose:
            _time = time() - _begin_time_yield
            print(f"Done. (Runtume: {_time:.4f} sec)")

        # Execute trades
        _begin_time_execute = time()
        for i, t in enumerate(trades):
            if verbose:
                print(f"\r{i + 1} trades executed: {t} ... ", end="")
            t.execute(universe)
        if verbose:
            _time = time() - _begin_time_execute
            print(f"Done. (Runtime: {_time:.4f} sec)")

        if verbose:
            _time = time() - _begin_time
            print(f"Done. (Runtime: {_time:.4f} sec)")

        self.trades = trades

        return self

    def get_logic(self):
        return getattr(self, "logic_func", self.logic)

    def get_params(self) -> dict:
        """
        Set the parameters of this strategy.

        Returns
        -------
        params : dict[str, *]
            Parameters.
        """
        return getattr(self, "params", {})

    def set_params(self, **params):
        """
        Set the parameters of this strategy.

        Parameters
        ----------
        - **params : dict
            Strategy parameters.

        Returns
        -------
        self : Strategy
            Strategy with new parameters.
        """
        valid_keys = self.get_params().keys()

        for key, value in params.items():
            if key not in valid_keys:
                raise ValueError(f"Invalid parameter: {key}")
            else:
                self.params[key] = value

        return self

    def score(self, metric_name) -> float:
        """
        Returns the value of a metric of self.

        Parameters
        ----------
        - metric_name : str
            Metric to evaluate.

        Returns
        -------
        metric_value : float
            Metric.
        """
        if not hasattr(self, "trades"):
            raise NotRunError("Strategy has not been run")

        return metric_from_name(metric_name)(self.trades, self.universe)

    def evaluate(self, metric):
        raise DeprecationWarning(
            "Strategy.evaluate(...) is deprecated. Use Strategy.score(...) instead."
        )
