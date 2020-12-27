import pytest

from itertools import cycle, islice

import numpy as np
from numpy.testing import assert_array_equal
from random import choice, choices

import epymetheus as ep


def yield_trades(n_orders):
    params_lot = [0.0, 1, 1.23, -1.23, 123.4, -123.4]

    if n_orders == 1:
        for lot in params_lot:
            yield ep.trade(
                asset="A0", entry="B0", exit="B1", lot=lot, take=1.0, stop=-1.0,
            )
    else:
        for i, _ in enumerate(params_lot):
            asset = [f"A{i}" for i in range(n_orders)]
            lot = list(islice(cycle(params_lot), i, i + n_orders))
            yield ep.trade(
                asset=asset, entry="B0", exit="B1", lot=lot, take=1.0, stop=-1.0,
            )


params_trade = list(yield_trades(1)) + list(yield_trades(2))
params_a = [0.0, 1, 1.23, -1.23, 123.4, -123.4]


def assert_trade_operation(trade0, trade1, operator):
    """
    Examples
    --------
    >>> trade0 = ep.trade(asset=['asset0', 'asset1'], lot=[1.0, -2.0])
    >>> trade1 = ep.trade(asset=['asset0', 'asset1'], lot=[2.0, -4.0])
    >>> operator = lambda x: 2 * x
    >>> assert_trade_operation(trade0, trade1, operator)  # No Error
    """
    assert_array_equal(trade0.asset, trade1.asset)
    assert trade0.entry == trade1.entry
    assert trade0.exit == trade1.exit
    assert np.allclose([operator(x) for x in trade0.array_lot], trade1.array_lot)
    assert trade0.take == trade1.take
    assert trade0.stop == trade1.stop


# --------------------------------------------------------------------------------


class TestOperation:
    @pytest.mark.parametrize("trade0", params_trade)
    @pytest.mark.parametrize("a", params_a)
    def test_mul(self, trade0, a):
        print(trade0)
        trade1 = a * trade0
        assert_trade_operation(trade0, trade1, lambda x: a * x)

    @pytest.mark.parametrize("trade0", params_trade)
    @pytest.mark.parametrize("a", params_a)
    def test_rmul(self, trade0, a):
        trade1 = trade0 * a
        assert_trade_operation(trade0, trade1, lambda x: a * x)

    @pytest.mark.parametrize("trade0", params_trade)
    def test_neg(self, trade0):
        trade1 = -trade0
        assert_trade_operation(trade0, trade1, lambda x: -x)

    @pytest.mark.parametrize("trade0", params_trade)
    @pytest.mark.parametrize("a", params_a)
    def test_truediv(self, trade0, a):
        if a != 0:
            trade1 = trade0 / a
            assert_trade_operation(trade0, trade1, lambda x: x / a)
        else:
            with pytest.raises(ZeroDivisionError):
                trade1 = trade0 / a


if __name__ == "__main__":
    from doctest import testmod

    testmod()
