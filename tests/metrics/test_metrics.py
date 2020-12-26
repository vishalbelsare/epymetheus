import pytest

import numpy as np
import pandas as pd

import epymetheus as ep
from epymetheus import Trade
from epymetheus.exceptions import NotRunError
from epymetheus.benchmarks import RandomStrategy, DeterminedStrategy
from epymetheus.datasets import make_randomwalk
from epymetheus.metrics import Return
from epymetheus.metrics import AverageReturn
from epymetheus.metrics import FinalWealth
from epymetheus.metrics import Drawdown
from epymetheus.metrics import MaxDrawdown
from epymetheus.metrics import Volatility
from epymetheus.metrics import SharpeRatio
from epymetheus.metrics import TradewiseSharpeRatio
from epymetheus.metrics import Exposure
from epymetheus.metrics import metric_from_name


# class TestBase:
#     """
#     Test common features of Metric.
#     """

#     @pytest.fixture(scope="function", autouse=True)
#     def setup(self):
#         np.random.seed(42)

#     @pytest.mark.parametrize("MetricClass", params_metric)
#     @pytest.mark.parametrize("seed", range(1))
#     def test_strategy_score(self, MetricClass, seed):
#         """
#         Test if `strategy.score(metric) == metric.result(strategy)`
#         """
#         m = MetricClass()
#         strategy = RandomStrategy(seed=seed).run(make_randomwalk())
#         result0 = np.array(m.result(strategy))  # from metric method
#         result1 = np.array(strategy.score(m))  # from strategy method
#         assert np.equal(result0, result1).all()

#     @pytest.mark.parametrize("MetricClass", params_metric)
#     @pytest.mark.parametrize("seed", range(1))
#     def test_call(self, MetricClass, seed):
#         """
#         Test if `metric.result(strategy) == metric(strategy)`.
#         """
#         m = MetricClass()
#         strategy = RandomStrategy(seed=seed).run(make_randomwalk())
#         result0 = np.array(m.result(strategy))  # from `result` method
#         result1 = np.array(m(strategy))  # from __call__
#         assert np.equal(result0, result1).all()

#     # @pytest.mark.parametrize("MetricClass", params_metric)
#     # def test_metric_from_name(self, MetricClass):
#     #     m = MetricClass()
#     #     assert metric_from_name(m.name).__class__ == m.__class__

#     def test_metric_from_name_nonexistent(self):
#         """
#         `_metric_from_name` is supposed to raise ValueError
#         when one gives it a nonexistent metric name.
#         """
#         with pytest.raises(KeyError):
#             metric_from_name("nonexistent_metric")

#     @pytest.mark.parametrize("MetricClass", params_metric)
#     def test_notrunerror(self, MetricClass):
#         """
#         Metric is supposed to raise NotRunError when one tries to score it
#         for a strategy which has not been run yet.
#         """
#         m = MetricClass()
#         with pytest.raises(NotRunError):
#             RandomStrategy(seed=42).score(m)


# class TestReturn:
#     """
#     Test if `Return` works as expected.
#     """

#     MetricClass = Return

#     @pytest.fixture(scope="function", autouse=True)
#     def setup(self):
#         np.random.seed(42)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("init_wealth", [100.0])
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_result_zero(self, rate, init_wealth, n_bars):
#         """
#         return = 0 for constant wealth.
#         """
#         series_wealth = init_wealth + np.zeros(n_bars)
#         result = Return(rate=rate)._result_from_wealth(series_wealth)
#         expected = np.zeros(n_bars)
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("rate", [True, False])
#     def test_result_hand(self, rate):
#         series_wealth = np.array([3, 1, 4, 1, 5, 9, 2], dtype=float)
#         result = Return(rate=rate)._result_from_wealth(series_wealth)
#         if rate:
#             expected = np.array([0, -2 / 3, 3 / 1, -3 / 4, 4 / 1, 4 / 5, -7 / 9])
#         else:
#             expected = np.array([0, -2, 3, -3, 4, 4, -7])
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("init_wealth", [100.0])
#     @pytest.mark.parametrize("seed", range(1))
#     def test_result_from_wealth(self, rate, init_wealth, seed):
#         """
#         `m._result_from_wealth(series_wealth) == m.result(strategy.wealth.wealth)`
#         """
#         m = self.MetricClass(rate=rate)
#         strategy = RandomStrategy(seed=seed).run(make_randomwalk())
#         series_wealth = init_wealth + strategy.wealth().values
#         result = m.result(strategy, init_wealth=init_wealth)
#         result_from_wealth = m._result_from_wealth(series_wealth)
#         assert np.allclose(result, result_from_wealth)


# class TestAverageReturn:
#     """
#     Test if `AverageReturn` works as expected.
#     """

#     MetricClass = AverageReturn

#     @pytest.fixture(scope="function", autouse=True)
#     def setup(self):
#         np.random.seed(42)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("n", [1, 365])
#     @pytest.mark.parametrize("init_wealth", [100.0])
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_result_zero(self, rate, n, init_wealth, n_bars):
#         """
#         average return = 0 for constant wealth.
#         """
#         series_wealth = init_wealth + np.zeros(n_bars)
#         result = self.MetricClass(rate=rate, n=n)._result_from_wealth(series_wealth)
#         expected = 0.0
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("n", [1, 2])  # can't be so large
#     def test_result_hand(self, rate, n):
#         series_wealth = np.array([3, 1, 4, 1, 5, 9, 2], dtype=float)
#         result = self.MetricClass(rate=rate, n=n)._result_from_wealth(series_wealth)
#         if rate:
#             expected = (2 / 3) ** (n / 6) - 1
#         else:
#             expected = -1 * (n / 6)
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("n", [1, 365])
#     @pytest.mark.parametrize("init_wealth", [10000.0])
#     def test_result_from_wealth(self, rate, n, init_wealth):
#         """
#         `m._result_from_wealth(series_wealth) == m.result(strategy.wealth.wealth)`
#         """
#         m = self.MetricClass(rate=rate, n=n)
#         strategy = RandomStrategy(seed=42).run(make_randomwalk())
#         series_wealth = init_wealth + strategy.wealth().values
#         result = m.result(strategy, init_wealth=init_wealth)
#         result_from_wealth = m._result_from_wealth(series_wealth)
#         assert np.allclose(result, result_from_wealth)


# class TestFinalWealth:
#     MetricClass = FinalWealth

#     @pytest.fixture(scope="function", autouse=True)
#     def setup(self):
#         np.random.seed(42)

#     @pytest.mark.parametrize("init_wealth", [0.0, 100.0])
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_result_zero(self, init_wealth, n_bars):
#         """
#         final wealth = initial wealth for zero return
#         """
#         series_wealth = init_wealth + np.zeros(n_bars, dtype=float)
#         result = self.MetricClass()._result_from_wealth(series_wealth)
#         expected = init_wealth
#         assert result == expected

#     @pytest.mark.parametrize("seed", range(1))
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_result_random(self, seed, n_bars):
#         np.random.seed(seed)
#         series_wealth = np.random.rand(n_bars)
#         result = self.MetricClass()._result_from_wealth(series_wealth)
#         expected = series_wealth[-1]
#         assert result == expected

#     @pytest.mark.parametrize("seed", range(1))
#     def test_result(self, seed):
#         m = self.MetricClass()
#         strategy = RandomStrategy(seed=seed).run(make_randomwalk())
#         result0 = m.result(strategy)
#         result1 = m._result_from_wealth(strategy.wealth().values)
#         assert result0 == result1


# class TestDrawdown:
#     """
#     Test `Drawdown`.
#     """

#     MetricClass = Drawdown

#     @pytest.fixture(scope="function", autouse=True)
#     def setup(self):
#         np.random.seed(42)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("init_wealth", [100.0])
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_result_zero(self, rate, init_wealth, n_bars):
#         series_wealth = init_wealth + np.zeros(n_bars)
#         result = self.MetricClass(rate=rate)._result_from_wealth(series_wealth)
#         expected = np.zeros(n_bars, dtype=float)
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("init_wealth", [100.0])
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_monotonous(self, rate, init_wealth, n_bars):
#         """
#         Drawdown = 0 for monotonously increasing wealth.
#         """
#         series_wealth = init_wealth + np.linspace(0.0, 100.0, n_bars)
#         result = self.MetricClass(rate=rate)._result_from_wealth(series_wealth)
#         expected = np.zeros(n_bars)
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("rate", [True, False])
#     def test_result_hand(self, rate):
#         series_wealth = np.array([3, 1, 4, 1, 5, 9, 2], dtype=float)
#         result = self.MetricClass(rate=rate)._result_from_wealth(series_wealth)
#         if rate:
#             expected = np.array([0, -2 / 3, 0, -3 / 4, 0, 0, -7 / 9], dtype=float)
#         else:
#             expected = np.array([0, -2, 0, -3, 0, 0, -7], dtype=float)
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("seed", range(1))
#     @pytest.mark.parametrize("rate", [True, False])
#     def test_result(self, seed, rate):
#         m = self.MetricClass(rate=rate)
#         strategy = RandomStrategy(seed=seed).run(make_randomwalk())

#         result0 = m.result(strategy)
#         result1 = m._result_from_wealth(strategy.wealth().values)

#         assert np.equal(result0, result1).all()


# class TestMaxDrawdown:
#     MetricClass = MaxDrawdown

#     @pytest.fixture(scope="function", autouse=True)
#     def setup(self):
#         np.random.seed(42)

#     def test_zero(self):
#         pass  # TODO

#     # @pytest.mark.parametrize("rate", [True, False])
#     # @pytest.mark.parametrize("init_wealth", [100.0])
#     # @pytest.mark.parametrize("n_bars", [100])
#     # def test_result_zero(self, rate, init_wealth, n_bars):
#     #     series_wealth = init_wealth + np.zeros(n_bars, dtype=float)

#     #     result = self.MetricClass(rate=rate)._result_from_wealth(series_wealth)
#     #     expected = 0

#     #     assert result == expected

#     # @pytest.mark.parametrize("rate", [True, False])
#     # @pytest.mark.parametrize("init_wealth", [100.0])
#     # @pytest.mark.parametrize("n_bars", [100])
#     # def test_monotonous(self, rate, init_wealth, n_bars):
#     #     """
#     #     Drawdown = 0 for monotonously increasing wealth.
#     #     """
#     #     series_wealth = init_wealth + np.linspace(0.0, 100.0, n_bars)
#     #     result = self.MetricClass(rate=rate)._result_from_wealth(series_wealth)
#     #     expected = 0
#     #     assert np.allclose(result, expected)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("seed", range(1))
#     @pytest.mark.parametrize("init_wealth", [10000.0])
#     def test_random(self, rate, seed, init_wealth):
#         strategy = RandomStrategy(seed=seed).run(make_randomwalk())
#         metric = self.MetricClass(rate=rate)
#         result = metric.result(strategy, init_wealth=init_wealth)
#         expected = np.min(Drawdown(rate=rate).result(strategy, init_wealth=init_wealth))
#         assert result == expected


# class TestVolatility:
#     MetricClass = Volatility

#     @pytest.fixture(scope="function", autouse=True)
#     def setup(self):
#         np.random.seed(42)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("init_wealth", [100.0])
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_result_zero(self, rate, init_wealth, n_bars):
#         series_wealth = init_wealth + np.zeros(n_bars)
#         result = self.MetricClass(rate=rate)._result_from_wealth(series_wealth)
#         expected = 0
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("init_wealth", [100.0])
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_result_monotonous(self, rate, init_wealth, n_bars):
#         series_wealth = init_wealth + np.linspace(0.0, 100.0, n_bars)

#         if rate:
#             series_wealth = np.exp(series_wealth)

#         result = self.MetricClass(rate=rate)._result_from_wealth(series_wealth)
#         expected = 0

#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("rate", [False])
#     def test_result_hand(self, rate):
#         series_wealth = np.array([3, 1, 4, 1, 4, 9, 2], dtype=float)

#         result = self.MetricClass(rate=rate)._result_from_wealth(series_wealth)
#         if rate:
#             expected = np.std([-2 / 3, 3 / 1, -3 / 4, 3 / 1, 5 / 5, -7 / 9])
#         else:
#             expected = np.std([-2, 3, -3, 3, 5, -7])
#         assert np.allclose(result, expected)


# class TestSharpeRatio:
#     """
#     Test `SharpeRatio`.
#     """

#     @pytest.fixture(scope="function", autouse=True)
#     def setup(self):
#         np.random.seed(42)

#     MetricClass = SharpeRatio

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("init_wealth", [100.0])
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_result_zero(self, rate, init_wealth, n_bars):
#         universe = pd.DataFrame(
#             {
#                 "A0": np.ones(n_bars, dtype=float),
#             }
#         )
#         strategy = DeterminedStrategy([ep.trade("A0")]).run(universe)
#         result = self.MetricClass(rate=rate).result(strategy, init_wealth=init_wealth)
#         expected = 0
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.mark.parametrize("n", [1, 365])
#     @pytest.mark.parametrize("risk_free_return", [0.0, 1.0])
#     @pytest.mark.parametrize("seed", range(1))
#     @pytest.mark.parametrize("init_wealth", [10000.0])
#     def test_random(self, rate, n, risk_free_return, seed, init_wealth):
#         strategy = RandomStrategy(seed=seed).run(make_randomwalk())
#         metric = self.MetricClass(rate=rate, n=n, risk_free_return=risk_free_return)
#         result = metric.result(strategy, init_wealth=init_wealth)
#         r = AverageReturn(rate=rate, n=n).result(strategy, init_wealth=init_wealth)
#         s = Volatility(rate=rate, n=n).result(strategy, init_wealth=init_wealth)
#         expected = (r - risk_free_return) / s
#         assert result == expected


# # class TestTradewiseSharpeRatio:
# #     """
# #     Test `TradewiseSharpeRatio`.
# #     """


# class TestExposure:
#     """
#     Test `Exposure`.
#     """

#     MetricClass = Exposure

#     universe_hand = pd.DataFrame(
#         {
#             "A0": [3, 1, 4, 1, 5, 9, 2],
#             "A1": [2, 7, 1, 8, 1, 8, 1],
#         }
#     )

#     @pytest.fixture(scope="function", autouse=True)
#     def setup(self):
#         np.random.seed(42)

#     @pytest.mark.parametrize("rate", [True, False])
#     @pytest.fixture(scope="function", autouse=True)
#     def setup(self):
#         np.random.seed(42)

#     @pytest.mark.parametrize("net", [True, False])
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_result_zero_0(self, net, n_bars):
#         universe = pd.DataFrame(
#             {
#                 "A0": np.zeros(n_bars, dtype=float),
#             }
#         )
#         strategy = DeterminedStrategy([ep.trade("A0")]).run(universe)
#         result = self.MetricClass(net=net).result(strategy)
#         expected = np.zeros(n_bars)
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("net", [True, False])
#     @pytest.mark.parametrize("n_bars", [100])
#     def test_result_zero_1(self, net, n_bars):
#         universe = pd.DataFrame(
#             {
#                 "A0": np.linspace(0.0, 1.0, n_bars),
#             }
#         )
#         strategy = DeterminedStrategy([ep.trade("A0", lot=0.0)]).run(universe)
#         result = self.MetricClass(net=net).result(strategy)
#         expected = np.zeros(n_bars)
#         assert np.allclose(result, expected)

#     @pytest.mark.parametrize("net", [True, False])
#     def test_hand(self, net):
#         universe = self.universe_hand
#         trade0 = ep.trade("A0", lot=2.0, open_bar=1, shut_bar=5)
#         trade1 = ep.trade("A1", lot=-3.0, open_bar=2, shut_bar=4)
#         strategy = DeterminedStrategy([trade0, trade1]).run(universe)
#         result = Exposure(net=net).result(strategy)
#         if net:
#             # 0 2  8   2 10 18 0
#             # 0 0 -3 -24 -3  0 0
#             expected = [0, 2, 5, -22, 7, 18, 0]
#         else:
#             expected = [0, 2, 11, 26, 13, 18, 0]

#         assert np.allclose(result, expected)
