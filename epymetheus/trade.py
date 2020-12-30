from copy import deepcopy

import numpy as np

from epymetheus.universe import Universe


def trade(
    asset,
    entry=None,
    exit=None,
    take=None,
    stop=None,
    lot=1.0,
    open_bar=None,
    shut_bar=None,
):
    """
    Initialize `Trade`.

    Parameters
    ----------
    - asset : str or array of str
        Name of assets.
    - entry : object or None, default None
        Datetime of entry.
    - exit : object or None, default None
        Datetime of exit.
    - take : float > 0 or None, default None
        Threshold of profit-take.
    - stop : float < 0 or None, default None
        Threshold of stop-loss.
    - lot : float, default 1.0
        Lot to trade in unit of share.

    Returns
    -------
    trade : Trade

    Examples
    --------
    >>> trade("AAPL")
    trade(['AAPL'], lot=[1.])

    >>> trade(["AAPL", "AMZN"])
    trade(['AAPL' 'AMZN'], lot=[1. 1.])

    >>> [1.0, -2.0] * trade(["AAPL", "AMZN"])
    trade(['AAPL' 'AMZN'], lot=[ 1. -2.])

    >>> from datetime import date
    >>> trade("AAPL", date(2020, 1, 1))
    trade(['AAPL'], lot=[1.], entry=2020-01-01)

    >>> trade("AAPL", date(2020, 1, 1), date(2020, 1, 31))
    trade(['AAPL'], lot=[1.], entry=2020-01-01, exit=2020-01-31)

    >>> trade("AAPL", take=200.0, stop=-100.0)
    trade(['AAPL'], lot=[1.], take=200.0, stop=-100.0)
    """
    if open_bar is not None:
        entry = open_bar if entry is None else entry
        raise DeprecationWarning("`open_bar` is deprecated. Use `entry` instead.")
    if shut_bar is not None:
        exit = shut_bar if exit is None else exit
        raise DeprecationWarning("`shut_bar` is deprecated. Use `exit` instead.")

    return Trade._trade(
        asset=asset, entry=entry, exit=exit, take=take, stop=stop, lot=lot
    )


class Trade:
    """
    Represent a single trade.

    Parameters
    ----------
    - asset : str or array of str
        Name of assets.
    - entry : object or None, default None
        Datetime of entry.
    - exit : object or None, default None
        Datetime of exit.
    - take : float > 0 or None, default None
        Threshold of profit-take.
    - stop : float < 0 or None, default None
        Threshold of stop-loss.
    - lot : float, default 1.0
        Lot to trade in unit of share.

    Attributes
    ----------
    - close: object
        Datetime to close the trade.
        It is set by the method `self.execute`.
    """

    def __init__(
        self, asset, entry=None, exit=None, take=None, stop=None, lot=1.0,
    ):
        self.asset = asset
        self.entry = entry
        self.exit = exit
        self.take = take
        self.stop = stop
        self.lot = lot

    @classmethod
    def _trade(cls, asset, entry, exit, take, stop, lot):
        """
        Initialize `Trade`.

        Returns
        -------
        trade : Trade
        """
        asset = np.asarray(asset).reshape(-1)
        lot = np.broadcast_to(np.asarray(lot), asset.shape)

        return cls(asset=asset, entry=entry, exit=exit, take=take, stop=stop, lot=lot)

    def execute(self, universe):
        """
        Execute trade and set `self.close`.

        Parameters
        ----------
        universe : pandas.DataFrame

        Returns
        -------
        self : Trade

        Examples
        --------
        >>> import pandas as pd
        >>> import epymetheus as ep
        >>> universe = pd.DataFrame({
        ...     "A0": [1., 2., 3., 4., 5., 6., 7.],
        ...     "A1": [2., 3., 4., 5., 6., 7., 8.],
        ...     "A2": [3., 4., 5., 6., 7., 8., 9.],
        ... }, dtype=float)

        >>> t = ep.trade("A0", entry=1, exit=6)
        >>> t = t.execute(universe)
        >>> t.close
        6

        >>> t = ep.trade("A0", entry=1, exit=6, take=2)
        >>> t = t.execute(universe)
        >>> t.close
        3

        >>> t = -ep.trade(asset="A0", entry=1, exit=6, stop=-2)
        >>> t = t.execute(universe)
        >>> t.close
        3
        """
        universe = self.__to_dataframe(universe)

        # If already executed
        if hasattr(self, "close"):
            return self

        entry = universe.index[0] if self.entry is None else self.entry
        exit = universe.index[-1] if self.exit is None else self.exit

        close = exit

        if (self.take is not None) or (self.stop is not None):
            i_entry = universe.index.get_indexer([entry]).item()
            i_exit = universe.index.get_indexer([exit]).item()

            # Compute pnl
            value = self.array_value(universe).sum(axis=1)
            pnl = value - value[i_entry]
            pnl[:i_entry] = 0

            # Place profit-take or loss-cut order
            take = self.take if self.take is not None else np.inf
            stop = self.stop if self.stop is not None else -np.inf
            signal = np.logical_or(pnl >= take, pnl <= stop)

            # Compute close
            i_order = np.searchsorted(signal, True)
            i_close = min(i_exit, i_order)
            close = universe.index[i_close]

        self.close = close

        return self

    def array_value(self, universe):
        """
        Return value of self for each asset.

        Returns
        -------
        array_value : numpy.array, shape (n_bars, n_orders)
            Array of values.

        Examples
        --------
        >>> import pandas as pd
        >>> import epymetheus as ep
        ...
        >>> universe = pd.DataFrame({
        ...     "A0": [1, 2, 3, 4, 5],
        ...     "A1": [2, 3, 4, 5, 6],
        ...     "A2": [3, 4, 5, 6, 7],
        ... })
        >>> trade = [2, -3] * ep.trade(["A0", "A2"], entry=1, exit=3)
        >>> trade.array_value(universe)
        array([[  2.,  -9.],
               [  4., -12.],
               [  6., -15.],
               [  8., -18.],
               [ 10., -21.]])
        """
        universe = self.__to_dataframe(universe)
        array_value = self.lot * universe.loc[:, self.asset].values
        return array_value

    def final_pnl(self, universe):
        """
        Return final profit-loss of self.

        Returns
        -------
        pnl : numpy.array, shape (n_orders, )

        Examples
        --------
        >>> import pandas as pd
        >>> import epymetheus as ep
        ...
        >>> universe = pd.DataFrame({
        ...     "A0": [1, 2, 3, 4, 5],
        ...     "A1": [2, 3, 4, 5, 6],
        ...     "A2": [3, 4, 5, 6, 7],
        ... }, dtype=float)
        >>> t = ep.trade(["A0", "A2"], entry=1, exit=3)
        >>> t = t.execute(universe)
        >>> t.final_pnl(universe)
        array([2., 2.])
        """
        universe = self.__to_dataframe(universe)

        i_entry = universe.index.get_indexer([self.entry]).item()
        i_close = universe.index.get_indexer([self.close]).item()

        value = self.array_value(universe)
        pnl = value - value[i_entry]
        pnl[:i_entry] = 0
        pnl[i_close:] = pnl[i_close]

        final_pnl = pnl[-1]

        return final_pnl

    def __eq__(self, other):
        attrs = ("asset", "entry", "exit", "take", "stop", "lot")
        return all(
            getattr(self, attr, None) == getattr(other, attr, None) for attr in attrs
        )

    def __mul__(self, num):
        return self.__rmul__(num)

    def __rmul__(self, num):
        """
        Multiply lot of self.

        Examples
        --------
        >>> trade("AMZN")
        trade(['AMZN'], lot=[1.])
        >>> (-2.0) * trade("AMZN")
        trade(['AMZN'], lot=[-2.])

        >>> trade(["AMZN", "AAPL"])
        trade(['AMZN' 'AAPL'], lot=[1. 1.])
        >>> (-2.0) * trade(["AMZN", "AAPL"])
        trade(['AMZN' 'AAPL'], lot=[-2. -2.])
        >>> [2.0, 3.0] * trade(["AMZN", "AAPL"])
        trade(['AMZN' 'AAPL'], lot=[2. 3.])
        """
        t = deepcopy(self)
        t.lot = t.lot * np.asarray(num)
        return t

    def __neg__(self):
        """
        Invert the lot of self.

        Examples
        --------
        >>> -trade("AMZN")
        trade(['AMZN'], lot=[-1.])
        """
        return (-1.0) * self

    def __truediv__(self, num):
        """
        Divide the lot of self.

        Examples
        --------
        >>> trade("AMZN", lot=2.0) / 2.0
        trade(['AMZN'], lot=[1.])

        >>> trade(["AMZN", "AAPL"], lot=[2.0, 4.0]) / 2.0
        trade(['AMZN' 'AAPL'], lot=[1. 2.])
        """
        return self.__mul__(1.0 / num)

    def __repr__(self):
        """
        >>> t = trade("AMZN", entry=1)
        >>> t
        trade(['AMZN'], lot=[1.], entry=1)

        >>> t = trade("AMZN", take=100.0)
        >>> t
        trade(['AMZN'], lot=[1.], take=100.0)
        """
        params = [f"{self.asset}", f"lot={self.lot}"]

        for attr in ("entry", "exit", "take", "stop"):
            value = getattr(self, attr)
            if value is not None:
                params.append(f"{attr}={value}")

        return f"trade({', '.join(params)})"

    @staticmethod
    def __to_dataframe(universe):
        # Backward compatibility
        return universe.prices if isinstance(universe, Universe) else universe
