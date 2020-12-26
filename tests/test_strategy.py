import pytest

import numpy as np
import pandas as pd
from epymetheus import trade
from epymetheus import create_strategy
from epymetheus import Strategy
from epymetheus.exceptions import NotRunError
from epymetheus.datasets import make_randomwalk
from epymetheus.benchmarks import RandomStrategy
from epymetheus.benchmarks import DeterminedStrategy


class MyStrategy(Strategy):
    def __init__(self, param_1, param_2):
        self.param_1 = param_1
        self.param_2 = param_2

    def logic(self, universe):
        yield (self.param_1 * trade("A"))
        yield (self.param_2 * trade("B"))


class TestStrategy:
    """
    Test `Strategy`
    """

    @staticmethod
    def my_strategy(universe, param_1, param_2):
        """
        Example logic
        """
        yield (param_1 * trade("A"))
        yield (param_2 * trade("B"))

    def test_init_from_func(self):
        strategy = create_strategy(self.my_strategy, param_1=1.0, param_2=2.0)
        universe = None

        assert strategy(universe) == [1.0 * trade("A"), 2.0 * trade("B")]

    def test_init_from_init(self):
        strategy = MyStrategy(param_1=1.0, param_2=2.0)
        universe = None

        assert strategy(universe) == [1.0 * trade("A"), 2.0 * trade("B")]

    def test_get_params(self):
        strategy = create_strategy(self.my_strategy, param_1=1.0, param_2=2.0)
        assert strategy.get_params() == {"param_1": 1.0, "param_2": 2.0}

        strategy = MyStrategy(param_1=1.0, param_2=2.0)
        assert strategy.get_params() == {}

    def test_set_params(self):
        strategy = create_strategy(self.my_strategy, param_1=1.0, param_2=2.0)
        assert strategy.get_params() == {"param_1": 1.0, "param_2": 2.0}
        strategy.set_params(param_1=3.0)
        assert strategy.get_params() == {"param_1": 3.0, "param_2": 2.0}
        with pytest.raises(ValueError):
            strategy.set_params(nonexistent_param=1.0)

    def test_warn(self):
        strategy = create_strategy(self.my_strategy, param_1=1.0, param_2=2.0)
        with pytest.raises(DeprecationWarning):
            strategy.evaluate(None)

    def test_sanity(self):
        np.random.seed(42)
        universe = make_randomwalk()
        strategy = RandomStrategy()

        strategy.run(universe)

        assert np.isclose(sum(strategy.history.pnl), strategy.wealth().values[-1])

    def test_history(self):
        universe = pd.DataFrame({"A": range(10), "B": range(10), "C": range(10)})
        trades = [
            trade("A", open_bar=1, shut_bar=2, take=3, stop=-4),
            [2, -3] * trade(["B", "C"], open_bar=3, shut_bar=9, take=5, stop=-2),
        ]
        strategy = DeterminedStrategy(trades).run(universe)
        history = strategy.history

        expected = pd.DataFrame(
            {
                "trade_id": [0, 1, 1],
                "asset": ["A", "B", "C"],
                "lot": [1, 2, -3],
                "open_bar": [1, 3, 3],
                "close_bar": [2, 5, 5],
                "shut_bar": [2, 9, 9],
                "take": [3, 5, 5],
                "stop": [-4, -2, -2],
                "pnl": [1, 4, -6],
            }
        )
        pd.testing.assert_frame_equal(history, expected, check_dtype=False)

    def test_history_notrunerror(self):
        strategy = RandomStrategy()
        with pytest.raises(NotRunError):
            # epymetheus.exceptions.NotRunError: Strategy has not been run
            strategy.history

    def test_wealth(self):
        # TODO test for when shut_bar != close_bar

        universe = pd.DataFrame({"A": range(10), "B": range(10), "C": range(10)})

        # TODO remove it when #129 will be fixed ---
        universe = universe.astype(float)
        # ---

        trades = [
            trade("A", open_bar=1, shut_bar=3),
            trade("B", open_bar=2, shut_bar=4),
        ]
        strategy = DeterminedStrategy(trades).run(universe)
        wealth = strategy.wealth()

        expected = pd.Series(
            [0, 0, 1, 3, 4, 4, 4, 4, 4, 4],
            index=universe.index,
        )

        pd.testing.assert_series_equal(wealth, expected, check_dtype=False)

    def test_wealth_notrunerror(self):
        strategy = RandomStrategy()
        with pytest.raises(NotRunError):
            # epymetheus.exceptions.NotRunError: Strategy has not been run
            strategy.wealth()


# import pytest

# import random
# import numpy as np

# from epymetheus import Trade, TradeStrategy
# from epymetheus.datasets import make_randomwalk
# from epymetheus.benchmarks import RandomTrader


# params_seed = [42]
# params_n_bars = [10, 1000]
# params_n_assets = [10, 100]
# params_n_trades = [10, 100]
# params_a = [1.23, -1.23]

# lots = [0.0, 1, 1.23, -1.23, 12345.678]


# class MultipleTradeStrategy(TradeStrategy):
#     """
#     Yield multiple trades.

#     Parameters
#     ----------
#     trades : iterable of Trade
#     """
#     def __init__(self, trades):
#         self.trades = trades

#     def logic(self, universe):
#         for trade in self.trades:
#             yield trade


# def make_random_trades(universe, n_trades, seed):
#     random_trader = RandomTrader(n_trades=n_trades, seed=seed)
#     trades = random_trader.run(universe).trades
#     return list(trades)  # for of array is slow


# def assert_add(history_0, history_1, history_A, attribute):
#     array_0 = getattr(history_0, attribute)
#     array_1 = getattr(history_1, attribute)
#     array_A = getattr(history_A, attribute)
#     array_01 = np.sort(np.concatenate([array_0, array_1]))
#     assert np.equal(array_01, np.sort(array_A)).all()


# def assert_mul(history_1, history_a, attribute, a=None):
#     array_1 = getattr(history_1, attribute)
#     array_a = getattr(history_a, attribute)
#     if a is not None:
#         array_1 *= float(a)

#     print(array_1, array_1.dtype)
#     print(array_a, array_a.dtype)

#     if array_1.dtype == np.float64:
#         assert np.allclose(array_1, array_a)
#     else:
#         assert (array_1 == array_a).all()


# # --------------------------------------------------------------------------------


# @pytest.mark.parametrize('seed', params_seed)
# @pytest.mark.parametrize('n_bars', params_n_bars)
# @pytest.mark.parametrize('n_assets', params_n_assets)
# @pytest.mark.parametrize('n_trades', params_n_trades)
# def test_linearity_add(seed, n_bars, n_assets, n_trades):
#     """
#     Test additivity of strategies for the following strategies:
#         - strategy_0 : yield (trade_00, trade_01, ...)
#         - strategy_1 : yield (trade_10, trade_11, ...)
#         - strategy_A : yield (trade_00, trade_01, ..., trade_10, trade_11, ...)
#     """
#     np.random.seed(seed)
#     random.seed(seed)

#     universe = make_randomwalk(n_bars, n_assets)

#     trades_0 = make_random_trades(universe, n_trades, seed + 0)
#     trades_1 = make_random_trades(universe, n_trades, seed + 1)
#     trades_A = trades_0 + trades_1

#     strategy_0 = MultipleTradeStrategy(trades=trades_0).run(universe)
#     strategy_1 = MultipleTradeStrategy(trades=trades_1).run(universe)
#     strategy_A = MultipleTradeStrategy(trades=trades_A).run(universe)

#     history_0 = strategy_0.history
#     history_1 = strategy_1.history
#     history_A = strategy_A.history

#     for attr in (
#         'asset',
#         'lot',
#         'open_bars',
#         'shut_bars',
#         'durations',
#         'open_prices',
#         'close_prices',
#         'gains',
#     ):
#         assert_add(history_0, history_1, history_A, attr)


# @pytest.mark.parametrize('seed', params_seed)
# @pytest.mark.parametrize('n_bars', params_n_bars)
# @pytest.mark.parametrize('n_assets', params_n_assets)
# @pytest.mark.parametrize('n_trades', params_n_trades)
# @pytest.mark.parametrize('a', params_a)
# def test_linearity_mul(seed, n_bars, n_assets, n_trades, a):
#     """
#     Test additivity of strategies for the following strategies:
#         - strategy_1 : yield (1 * trade_0, 1 * trade_11, ...)
#         - strategy_a : yield (a * trade_0, a * trade_01, ...)
#     """
#     np.random.seed(seed)
#     random.seed(seed)

#     universe = make_randomwalk(n_bars, n_assets)

#     trades_1 = make_random_trades(universe, n_trades, seed + 1)
#     trades_a = [
#         Trade(
#             asset=trade.asset,
#             lot=a * trade.lot,
#             open_bar=trade.open_bar,
#             shut_bar=trade.shut_bar
#         )
#         for trade in trades_1
#     ]

#     strategy_1 = MultipleTradeStrategy(trades=trades_1).run(universe)
#     strategy_a = MultipleTradeStrategy(trades=trades_a).run(universe)

#     history_1 = strategy_1.history
#     history_a = strategy_a.history

#     for attr in (
#         'asset',
#         'open_bars',
#         'shut_bars',
#         'durations',
#         'open_prices',
#         'close_prices',
#     ):
#         assert_mul(history_1, history_a, attr, None)

#     for attr in ('lot', 'gains'):
#         assert_mul(history_1, history_a, attr, a)
