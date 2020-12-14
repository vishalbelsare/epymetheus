from copy import deepcopy

import numpy as np

from epymetheus.universe import Universe


def trade(
    asset,
    lot=1.0,
    open_bar=None,
    shut_bar=None,
    take=None,
    stop=None,
):
    """
    Initialize `Trade`.

    Parameters
    ----------
    - asset : str or array of str
        Name of assets.
    - open_bar : object or None, default None
        Bar to open the trade.
    - shut_bar : object or None, default None
        Bar to enforce the trade to close.
    - lot : float, default 1.0
        Lot to trade in unit of share.
    - take : float > 0 or None, default None
        Threshold of profit-take.
    - stop : float < 0 or None, default None
        Threshold of stop-loss.

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
    >>> trade("AAPL", open_bar=date(2020, 1, 1))
    trade(['AAPL'], lot=[1.], open_bar=2020-01-01)

    >>> trade("AAPL", open_bar=date(2020, 1, 1), shut_bar=date(2020, 1, 31))
    trade(['AAPL'], lot=[1.], open_bar=2020-01-01, shut_bar=2020-01-31)

    >>> trade("AAPL", take=200.0, stop=-100.0)
    trade(['AAPL'], lot=[1.], take=200.0, stop=-100.0)
    """
    return Trade._trade(
        asset=asset,
        lot=lot,
        open_bar=open_bar,
        shut_bar=shut_bar,
        take=take,
        stop=stop,
    )


class Trade:
    """
    Represent a single trade.

    Parameters
    ----------
    - asset : str or array of str
        Name of assets.
    - open_bar : object or None, default None
        Bar to open the trade.
    - shut_bar : object or None, default None
        Bar to enforce the trade to close.
    - lot : float, default 1.0
        Lot to trade in unit of share.
    - take : float > 0 or None, default None
        Threshold of profit-take.
    - stop : float < 0 or None, default None
        Threshold of stop-loss.

    Attributes
    ----------
    - close_bar : object
        Bar to close the trade.
        It is set by the method `self.execute`.
    """

    def __init__(
        self,
        asset,
        lot=1.0,
        open_bar=None,
        shut_bar=None,
        take=None,
        stop=None,
    ):
        self.asset = asset
        self.lot = lot
        self.open_bar = open_bar
        self.shut_bar = shut_bar
        self.take = take
        self.stop = stop

    @classmethod
    def _trade(
        cls,
        asset,
        lot=1.0,
        open_bar=None,
        shut_bar=None,
        take=None,
        stop=None,
    ):
        """
        Initialize `Trade`.

        Returns
        -------
        trade : Trade
        """
        asset = np.asarray(asset).reshape(-1)
        lot = np.broadcast_to(np.asarray(lot), asset.shape)

        return cls(
            asset=asset,
            lot=lot,
            open_bar=open_bar,
            shut_bar=shut_bar,
            take=take,
            stop=stop,
        )

    @property
    def array_asset(self):
        """
        Return asset as `numpy.array`.

        Returns
        -------
        array_asset : numpy.array, shape (n_orders, )

        Examples
        --------
        >>> trade = Trade(asset='AAPL')
        >>> trade.array_asset
        array(['AAPL'], dtype='<U4')

        >>> trade = Trade(asset=['AAPL', 'MSFT'])
        >>> trade.array_asset
        array(['AAPL', 'MSFT'], dtype='<U4')
        """
        return np.asarray(self.asset).reshape(-1)

    @property
    def array_lot(self):
        """
        Return lot as `numpy.array`.

        Returns
        -------
        array_lot : numpy.array, shape (n_orders, )

        Examples
        --------
        >>> trade = Trade(asset='AAPL', lot=0.2)
        >>> trade.array_lot
        array([0.2])
        >>> trade = Trade(asset=['AAPL', 'MSFT'], lot=[0.2, 0.4])
        >>> trade.array_lot
        array([0.2, 0.4])
        """
        return np.asarray(self.lot).reshape(-1)

    @property
    def n_orders(self):
        """
        Return number of assets in self.

        Returns
        -------
        n_orders : int
            Number of orders.

        Examples
        --------
        >>> trade = Trade(asset='AAPL')
        >>> trade.n_orders
        1
        >>> trade = Trade(asset=['AAPL', 'MSFT'])
        >>> trade.n_orders
        2
        """
        return self.array_asset.size

    def execute(self, universe):
        """
        Execute trade and set `self.close_bar`.

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
        ...     "A0": [1, 2, 3, 4, 5, 6, 7],
        ...     "A1": [2, 3, 4, 5, 6, 7, 8],
        ...     "A2": [3, 4, 5, 6, 7, 8, 9],
        ... }, dtype=float)

        >>> t = ep.trade("A0", open_bar=1, shut_bar=6)
        >>> t = t.execute(universe)
        >>> t.close_bar
        6

        >>> t = ep.trade("A0", open_bar=1, shut_bar=6, take=2)
        >>> t = t.execute(universe)
        >>> t.close_bar
        3

        >>> t = -ep.trade(asset="A0", open_bar=1, shut_bar=6, stop=-2)
        >>> t = t.execute(universe)
        >>> t.close_bar
        3
        """
        universe = self.__to_dataframe(universe)

        # If already executed
        if hasattr(self, "close_bar"):
            return self

        # Compute close_bar
        open_bar = universe.index[0] if self.open_bar is None else self.open_bar
        shut_bar = universe.index[-1] if self.shut_bar is None else self.shut_bar

        close_bar = shut_bar

        if (self.take is not None) or (self.stop is not None):
            i_open = universe.index.get_indexer([open_bar]).item()
            i_shut = universe.index.get_indexer([shut_bar]).item()

            value = self.array_value(universe).sum(axis=1)
            pnl = value - value[i_open]
            pnl[:i_open] = 0

            signal = np.logical_or(
                pnl >= (self.take or np.inf),
                pnl <= (self.stop or -np.inf),
            )
            i_signal = np.searchsorted(signal, True)

            i_close = min(i_shut, i_signal)
            close_bar = universe.index[i_close]

        self.close_bar = close_bar

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
        >>> trade = [2, -3] * ep.trade(["A0", "A2"], open_bar=1, shut_bar=3)
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

    # def array_exposure(self, universe):
    #     """
    #     Return exposure of self for each order.

    #     Returns
    #     -------
    #     array_exposure : numpy.array, shape (n_bars, n_orders)

    #     Examples
    #     --------
    #     >>> import pandas as pd
    #     >>> import epymetheus as ep
    #     ...
    #     >>> universe = pd.DataFrame({
    #     ...     "A0": [1, 2, 3, 4, 5],
    #     ...     "A1": [2, 3, 4, 5, 6],
    #     ...     "A2": [3, 4, 5, 6, 7],
    #     ... }, dtype=float)
    #     >>> trade = [2, -3] * ep.trade(["A0", "A2"], open_bar=1, shut_bar=3)
    #     >>> trade.array_exposure(universe)
    #     array([[  0.,   0.],
    #            [  4., -12.],
    #            [  6., -15.],
    #            [  8., -18.],
    #            [  0.,   0.]])
    #     """
    #     universe = self.__to_dataframe(universe)

    #     i_open= universe.index.get_indexer([self.open_bar]).item()
    #     i_close= universe.index.get_indexer([self.close_bar]).item()

    #     value = self.array_value(universe)
    #     value[:open_bar_index] = 0
    #     value[stop_bar_index + 1 :] = 0

    #     array_exposure = array_exposure.reshape(-1, self.asset.size)

    #     return array_exposure

    # def series_exposure(self, universe, net=True):
    #     """
    #     Return time-series of value of the position.

    #     Parameters
    #     ----------
    #     - universe : pandas.DataFram e
    #     - net : bool, default True
    #         If True, return net exposure.
    #         If False, return absolute exposure.

    #     Returns
    #     -------
    #     series_exposure : numpy.array, shape (n_bars, )

    #     Examples
    #     --------
    #     >>> import pandas as pd
    #     >>> import epymetheus as ep
    #     ...
    #     >>> universe = pd.DataFrame({
    #     ...     "A0": [1, 2, 3, 4, 5],
    #     ...     "A1": [2, 3, 4, 5, 6],
    #     ...     "A2": [3, 4, 5, 6, 7],
    #     ... })
    #     >>> t = [2, -3] * ep.trade(["A0", "A2"], open_bar=1, shut_bar=3)
    #     >>> t.series_exposure(universe, net=True)
    #     array([  0.,  -8.,  -9., -10.,   0.])
    #     >>> t.series_exposure(universe, net=False)
    #     array([ 0., 16., 21., 26.,  0.])
    #     """
    #     universe = self.__to_dataframe(universe)

    #     array_exposure = self.array_exposure(universe)
    #     if net:
    #         series_exposure = array_exposure.sum(axis=1)
    #     else:
    #         series_exposure = np.abs(array_exposure).sum(axis=1)

    #     return series_exposure

    def array_pnl(self, universe):
        """
        Return profit-loss of self for each order.

        Returns
        -------
        array_pnl : numpy.array, shape (n_bars, n_orders)

        Examples
        --------
        >>> import pandas as pd
        >>> import epymetheus as ep
        >>> universe = pd.DataFrame({
        ...     "A0": [1, 2, 3, 4, 5],
        ...     "A1": [3, 4, 5, 6, 7],
        ... })
        >>> trade = [2, -3] * ep.trade(["A0", "A1"], open_bar=1, shut_bar=3)
        >>> trade.array_pnl(universe)
        array([[ 0.,  0.],
               [ 0.,  0.],
               [ 2., -3.],
               [ 4., -6.],
               [ 4., -6.]])
        """
        universe = self.__to_dataframe(universe)

        array_value = self.array_value(universe)

        stop_bar = universe.index[-1] if self.shut_bar is None else self.shut_bar

        open_bar_index = universe.index.get_indexer([self.open_bar]).item()
        stop_bar_index = universe.index.get_indexer([stop_bar]).item()

        array_pnl = array_value
        array_pnl -= array_pnl[open_bar_index]
        array_pnl[:open_bar_index] = 0
        array_pnl[stop_bar_index:] = array_pnl[stop_bar_index]

        array_pnl = array_pnl.reshape(-1, self.asset.size)

        return array_pnl

    def series_pnl(self, universe):
        """
        Return profit-loss of self.

        Returns
        -------
        net_exposure : numpy.array, shape (n_bars, )

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
        >>> t = ep.trade("A0", lot=1, open_bar=1, shut_bar=3)
        >>> t = t.execute(universe)
        >>> t.series_pnl(universe)
        array([0, 0, 1, 2, 2])
        """
        universe = self.__to_dataframe(universe)

        return self.array_pnl(universe).sum(axis=1)

    def final_pnl(self, universe):
        """
        Return final profit-loss of self.

        Returns
        -------
        pnl : numpy.array, shapr (n_orders, )

        Raises
        ------
        ValueError
            If self has not been `run`.

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
        >>> t = ep.trade(["A0", "A2"], open_bar=1, shut_bar=3)
        >>> t = t.execute(universe)
        >>> t.final_pnl(universe)
        array([2., 2.])
        """
        universe = self.__to_dataframe(universe)

        i_open = universe.index.get_indexer([self.open_bar]).item()
        i_close = universe.index.get_indexer([self.close_bar]).item()

        value = self.array_value(universe)
        pnl = value - value[i_open]
        pnl[:i_open] = 0
        pnl[i_close:] = pnl[i_close]

        final_pnl = pnl[-1]

        return final_pnl

    def __eq__(self, other):
        attrs = (
            "asset",
            "lot",
            "open_bar",
            "shut_bar",
            "take",
            "stop",
        )
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
        >>> t = trade("AMZN", open_bar=1)
        >>> t
        trade(['AMZN'], lot=[1.], open_bar=1)

        >>> t = trade("AMZN", take=100.0)
        >>> t
        trade(['AMZN'], lot=[1.], take=100.0)
        """
        params = [f"{self.asset}", f"lot={self.lot}"]

        for attr in ("open_bar", "shut_bar", "take", "stop"):
            value = getattr(self, attr)
            if value is not None:
                params.append(f"{attr}={value}")

        return f"trade({', '.join(params)})"

    @staticmethod
    def __to_dataframe(universe):
        # Backward compatibility
        return universe.prices if isinstance(universe, Universe) else universe
