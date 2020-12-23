import pytest

import numpy as np
import pandas as pd

import epymetheus as ep
from epymetheus import Trade
from epymetheus.datasets import make_randomwalk
from epymetheus.benchmarks import RandomStrategy


class TestInit:
    """
    Test initialization by function `trade`.
    """

    def test_shape(self):
        trade = ep.trade("A", lot=1.0)
        assert trade.asset.shape == (1,)
        assert trade.lot.shape == (1,)

        trade = ep.trade(["A"], lot=1.0)
        assert trade.asset.shape == (1,)
        assert trade.lot.shape == (1,)

        trade = ep.trade("A", lot=[1.0])
        assert trade.asset.shape == (1,)
        assert trade.lot.shape == (1,)

        trade = ep.trade(["A", "B"], lot=1.0)
        assert trade.asset.shape == (2,)
        assert trade.lot.shape == (2,)

        trade = ep.trade(["A", "B"], lot=[1.0])
        assert trade.asset.shape == (2,)
        assert trade.lot.shape == (2,)

        trade = ep.trade(["A", "B"], lot=[1.0, 2.0])
        assert trade.asset.shape == (2,)
        assert trade.lot.shape == (2,)


class TestExecute:
    pass


class TestArrayValue:
    """
    Test `Trade.array_value()`.
    """

    universe_hand = pd.DataFrame(
        {
            "A0": [3, 1, 4, 1, 5, 9, 2],
            "A1": [2, 7, 1, 8, 2, 8, 1],
        },
        index=range(7),
        dtype=float,
    )
    trade0 = ep.trade(["A0", "A1"], lot=[2, -3], open_bar=1, shut_bar=3)
    trade1 = ep.trade(["A1", "A0"], lot=[-3, 2], open_bar=1, shut_bar=3)
    expected0 = [[6, -6], [2, -21], [8, -3], [2, -24], [10, -6], [18, -24], [4, -3]]
    expected1 = [[-6, 6], [-21, 2], [-3, 8], [-24, 2], [-6, 10], [-24, 18], [-3, 4]]

    @pytest.mark.parametrize(
        "trade, expected",
        [(trade0, expected0), (trade1, expected1)],
    )
    def test_value_hand(self, trade, expected):
        result = trade.array_value(universe=self.universe_hand)
        assert np.allclose(result, expected)

    def test_value_zero(self):
        trade = ep.trade(asset=["A0", "A1"], lot=[0, 0], open_bar=1, shut_bar=3)
        result = trade.array_value(self.universe_hand)
        expected = np.zeros((len(self.universe_hand.index), 2))
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("seed", range(1))
    def test_linearity_add(self, seed):
        np.random.seed(seed)
        lot0, lot1 = np.random.random(2)
        trade0 = ep.trade("A0", lot=lot0, open_bar=1, shut_bar=3)
        trade1 = ep.trade("A0", lot=lot1, open_bar=1, shut_bar=3)
        tradeA = ep.trade("A0", lot=lot0 + lot1, open_bar=1, shut_bar=3)
        result0 = trade0.array_value(self.universe_hand)
        result1 = trade1.array_value(self.universe_hand)
        resultA = tradeA.array_value(self.universe_hand)
        assert np.allclose(result0 + result1, resultA)

    @pytest.mark.parametrize("a", [-2.0, -1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("seed", range(1))
    def test_linearity_mul(self, a, seed):
        np.random.seed(seed)
        universe = make_randomwalk()
        trade0 = RandomStrategy(n_trades=1, seed=seed).run(universe).trades[0]
        tradeA = a * trade0
        print(trade0, tradeA)
        result0 = trade0.array_value(universe)
        resultA = tradeA.array_value(universe)
        assert np.allclose(a * result0, resultA)


# class TestArrayExposure:
#     """
#     Test `Trade.array_exposure()`.
#     """

#     universe_hand = pd.DataFrame(
#         {
#             "A0": [3, 1, 4, 1, 5, 9, 2],
#             "A1": [2, 7, 1, 8, 2, 8, 1],
#         },
#         index=range(7),
#         dtype=float,
#     )

#     trade0 = ep.trade(asset=["A0", "A1"], lot=[2, -3], open_bar=1, shut_bar=3)
#     trade1 = ep.trade(asset=["A1", "A0"], lot=[-3, 2], open_bar=1, shut_bar=3)
#     expected0 = [[0, 0], [2, -21], [8, -3], [2, -24], [0, 0], [0, 0], [0, 0]]
#     expected1 = [[0, 0], [-21, 2], [-3, 8], [-24, 2], [0, 0], [0, 0], [0, 0]]

#     @pytest.mark.parametrize(
#         "trade, expected",
#         [(trade0, expected0), (trade1, expected1)],
#     )
#     def test_hand(self, trade, expected):
#         result = trade.array_exposure(universe=self.universe_hand)
#         assert np.allclose(result, expected)

#     def test_value_zero(self):
#         trade = ep.trade(asset=["A0", "A1"], lot=[0, 0], open_bar=1, shut_bar=3)
#         result = trade.array_exposure(self.universe_hand)
#         expected = np.zeros((len(self.universe_hand.index), 2))
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("seed", range(1))
#     def test_linearity_add(self, seed):
#         np.random.seed(seed)
#         lot0, lot1 = np.random.random(2)
#         trade0 = ep.trade(asset="A0", lot=lot0, open_bar=1, shut_bar=3)
#         trade1 = ep.trade(asset="A0", lot=lot1, open_bar=1, shut_bar=3)
#         tradeA = ep.trade(asset="A0", lot=lot0 + lot1, open_bar=1, shut_bar=3)
#         result0 = trade0.array_exposure(self.universe_hand)
#         result1 = trade1.array_exposure(self.universe_hand)
#         resultA = tradeA.array_exposure(self.universe_hand)
#         assert np.allclose(result0 + result1, resultA)

#     @pytest.mark.parametrize("a", [-2.0, -1.0, 0.0, 1.0, 2.0])
#     @pytest.mark.parametrize("seed", range(1))
#     def test_linearity_mul(self, a, seed):
#         np.random.seed(seed)
#         universe = make_randomwalk()
#         trade0 = RandomStrategy(n_trades=1, seed=seed).run(universe).trades[0]
#         tradeA = a * trade0
#         result0 = trade0.array_exposure(universe)
#         resultA = tradeA.array_exposure(universe)
#         assert np.allclose(a * result0, resultA)


# class TestSeriesExposure:
#     """
#     Test `Trade.series_exposure()`.
#     """

#     universe_hand = pd.DataFrame(
#         {
#             "A0": [3, 1, 4, 1, 5, 9, 2],
#             "A1": [2, 7, 1, 8, 2, 8, 1],
#         },
#         index=range(7),
#         dtype=float,
#     )

#     @pytest.mark.parametrize("net", [True, False])
#     def test_hand(self, net):
#         trade = ep.trade(asset=["A0", "A1"], lot=[2, -3], open_bar=1, shut_bar=3)
#         result = trade.series_exposure(net=net, universe=self.universe_hand)
#         if net:
#             expected = [0, -19, 5, -22, 0, 0, 0]
#         else:
#             expected = [0, 23, 11, 26, 0, 0, 0]

#         assert np.allclose(result, expected)

#     def test_value_zero(self):
#         trade = ep.trade(asset=["A0", "A1"], lot=[0, 0], open_bar=1, shut_bar=3)
#         result = trade.series_exposure(self.universe_hand)
#         expected = np.zeros((len(self.universe_hand.index)))
#         assert np.allclose(result, expected)

#     # Abs exposure doesn't satisfy linearity add
#     @pytest.mark.parametrize("net", [True])
#     @pytest.mark.parametrize("seed", range(1))
#     def test_linearity_add(self, net, seed):
#         np.random.seed(seed)
#         lot0, lot1 = np.random.random(2)
#         trade0 = ep.trade(asset="A0", lot=lot0, open_bar=1, shut_bar=3)
#         trade1 = ep.trade(asset="A0", lot=lot1, open_bar=1, shut_bar=3)
#         tradeA = ep.trade(asset="A0", lot=lot0 + lot1, open_bar=1, shut_bar=3)
#         result0 = trade0.series_exposure(self.universe_hand, net=net)
#         result1 = trade1.series_exposure(self.universe_hand, net=net)
#         resultA = tradeA.series_exposure(self.universe_hand, net=net)
#         assert np.allclose(result0 + result1, resultA)

#     @pytest.mark.parametrize("a", [-2.0, -1.0, 0.0, 1.0, 2.0])
#     @pytest.mark.parametrize("net", [True, False])
#     @pytest.mark.parametrize("seed", range(1))
#     def test_linearity_mul(self, a, net, seed):
#         np.random.seed(42)
#         universe = make_randomwalk()
#         trade0 = RandomStrategy(n_trades=1, seed=seed).run(universe).trades[0]
#         tradeA = a * trade0
#         result0 = trade0.series_exposure(universe, net=net)
#         resultA = tradeA.series_exposure(universe, net=net)
#         if net:
#             assert np.allclose(a * result0, resultA)
#         else:
#             assert np.allclose(abs(a) * result0, resultA)


class TestArrayPnl:
    """
    Test `Trade.series_pnl()`.
    """

    # TODO


class TestSeriesPnl:
    """
    Test `Trade.series_pnl()`.
    """

    universe_hand = pd.DataFrame(
        {
            "A0": [3, 1, 4, 1, 5, 9, 2],
            "A1": [2, 7, 1, 8, 2, 8, 1],
        },
        index=range(7),
        dtype=float,
    )
    trade0 = ep.trade(asset=["A0", "A1"], lot=[2, -3], open_bar=1, shut_bar=3)
    expected0 = [0, 0, 24, -3, -3, -3, -3]

    @pytest.mark.parametrize("trade, expected", [(trade0, expected0)])
    def test_value_hand(self, trade, expected):
        result = trade.series_pnl(universe=self.universe_hand)
        assert np.allclose(result, expected)

    def test_value_zero(self):
        trade = ep.trade(asset=["A0", "A1"], lot=[0, 0], open_bar=1, shut_bar=3)
        result = trade.series_pnl(self.universe_hand)
        expected = np.zeros(len((self.universe_hand.index)))
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("seed", range(1))
    def test_linearity_add(self, seed):
        np.random.seed(seed)
        lot0, lot1 = np.random.random(2)
        trade0 = ep.trade(asset="A0", lot=lot0, open_bar=1, shut_bar=3)
        trade1 = ep.trade(asset="A0", lot=lot1, open_bar=1, shut_bar=3)
        tradeA = ep.trade(asset="A0", lot=lot0 + lot1, open_bar=1, shut_bar=3)
        result0 = trade0.series_pnl(self.universe_hand)
        result1 = trade1.series_pnl(self.universe_hand)
        resultA = tradeA.series_pnl(self.universe_hand)
        assert np.allclose(result0 + result1, resultA)

    @pytest.mark.parametrize("a", [-2.0, -1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("seed", range(1))
    def test_linearity_mul(self, a, seed):
        np.random.seed(42)
        universe = make_randomwalk()
        trade0 = RandomStrategy(n_trades=1, seed=seed).run(universe).trades[0]
        tradeA = a * trade0
        result0 = trade0.series_pnl(universe)
        resultA = tradeA.series_pnl(universe)
        assert np.allclose(a * result0, resultA)


class TestFinalPnl:
    universe_hand = pd.DataFrame(
        {
            "A0": [3, 1, 4, 1, 5, 9, 2],
            "A1": [2, 7, 1, 8, 2, 8, 1],
        },
        index=range(7),
        dtype=float,
    )

    @pytest.mark.parametrize("seed", range(10))
    def test_random(self, seed):
        np.random.seed(42)
        universe = make_randomwalk(n_bars=100, n_assets=10)
        trade = RandomStrategy(n_trades=1, seed=seed).run(universe).trades[0]
        result = trade.final_pnl(universe)
        expected = trade.array_pnl(universe)[-1]

        assert np.allclose(result, expected)

    @pytest.mark.parametrize("a", [-2.0, -1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("seed", range(1))
    def test_linearity_mul(self, a, seed):
        np.random.seed(42)
        universe = make_randomwalk()
        trade0 = RandomStrategy(n_trades=1, seed=seed).run(universe).trades[0]
        tradeA = a * trade0
        result0 = trade0.final_pnl(universe)
        resultA = tradeA.final_pnl(universe)
        assert np.allclose(a * result0, resultA)


class TestRepr:
    """
    Test `ep.trade.__repr__`.
    """

    def test_repr(self):
        trade = ep.trade("A")
        assert repr(trade) == "trade(['A'], lot=[1.])"

        trade = ep.trade("A", lot=2, take=3.0, stop=-3.0, open_bar="B0", shut_bar="B1")

        assert (
            repr(trade)
            == "trade(['A'], lot=[2], open_bar=B0, shut_bar=B1, take=3.0, stop=-3.0)"
        )


# @pytest.mark.parametrize("seed", params_seed)
# def test_execute_0_0(seed):
#     """
#     Test `trade.execute` sets `trade.close_bar` correctly.

#     Setup
#     -----
#     - trade.take is None
#     - trade.stop is None
#     - trade.shut_bar is not None

#     Expected Result
#     ---------------
#     trade.close_bar == universe.shut_bar
#     """
#     # shut_bar is not None
#     universe = make_randomwalk(seed=seed)
#     trade = make_random_trade(universe, seed=seed)
#     trade.execute(universe)

#     assert trade.close_bar == trade.shut_bar


# @pytest.mark.parametrize("seed", params_seed)
# def test_execute_0_1(seed):
#     """
#     Test `trade.execute` sets `trade.close_bar` correctly.

#     Setup
#     -----
#     - trade.take is None
#     - trade.stop is None
#     - trade.shut_bar is None

#     Expected Result
#     ---------------
#     trade.close_bar == universe.bars[-1]
#     """
#     # shut_bar is not None
#     universe = make_randomwalk(seed=seed)
#     trade = make_random_trade(universe, seed=seed)
#     trade.shut_bar = None
#     trade.execute(universe)

#     assert trade.close_bar == universe.bars[-1]


# @pytest.mark.parametrize("seed", params_seed)
# def test_execute(seed):
#     """
#     Test `trade.execute` sets `trade.close_bar` correctly.

#     Setup
#     -----
#     - trade.take is None
#     - trade.stop is None
#     - trade.shut_bar is None

#     Expected Result
#     ---------------
#     trade.close_bar == universe.bars[-1]
#     """
#     # shut_bar is not None
#     universe = make_randomwalk(seed=seed)
#     trade = make_random_trade(universe, seed=seed)
#     trade.shut_bar = None
#     trade.execute(universe)

#     assert trade.close_bar == universe.bars[-1]


# # @pytest.mark.parametrize('seed', params_seed)
# # @pytest.mark.parametrize('n_bars', params_n_bars)
# # @pytest.mark.parametrize('const', params_const)
# # def test_execute(seed, n_bars, const):
# #     period = n_samples // 10
# #     shift = np.random.randint(period)
# #     prices = pd.DataFrame({
# #         'Asset0': const + make_sin(n_bars=n_bars, period=period, shift=shift)
# #     })
# #     universe = prices

# #     trade = ep.trade('Asset0', lot=1.0, )


# # def test_execute_take():
# #     universe = pd.DataFrame({"Asset0": np.arange(100, 200)})

# #     trade = ep.trade("Asset0", lot=1.0, take=1.9, open_bar=1, shut_bar=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 3
# #     assert np.array_equal(trade.final_pnl(universe), [103 - 101])

# #     trade = ep.trade("Asset0", lot=2.0, take=3.8, open_bar=1, shut_bar=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 3
# #     assert np.array_equal(trade.final_pnl(universe), [2 * (103 - 101)])

# #     trade = ep.trade("Asset0", lot=1.0, take=1000, open_bar=1, shut_bar=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 5
# #     assert np.array_equal(trade.final_pnl(universe), [105 - 101])


# # def test_execute_stop():
# #     universe = prices=pd.DataFrame({"Asset0": np.arange(100, 0, -1)})

# #     trade = ep.trade("Asset0", lot=1.0, stop=-1.9, open_bar=1, shut_bar=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 3
# #     assert np.array_equal(trade.final_pnl(universe), [97 - 99])

# #     trade = ep.trade("Asset0", lot=2.0, stop=-3.8, open_bar=1, shut_bar=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 3
# #     assert np.array_equal(trade.final_pnl(universe), [2 * (97 - 99)])

# #     trade = ep.trade("Asset0", lot=1.0, stop=-1000, open_bar=1, shut_bar=5)
# #     trade.execute(universe)
# #     assert trade.close_bar == 5
# #     assert np.array_equal(trade.final_pnl(universe), [95 - 99])


# # TODO both take and stop
# # TODO short position
# # TODO multiple orders

# # def test_execute_takestop():
# #     pass
