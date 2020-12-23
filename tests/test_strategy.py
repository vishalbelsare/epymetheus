import pytest

import numpy as np
from epymetheus import trade
from epymetheus import create_strategy
from epymetheus import Strategy

from epymetheus.datasets import make_randomwalk
from epymetheus.benchmarks import RandomStrategy


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
