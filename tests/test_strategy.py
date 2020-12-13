import pytest

from epymetheus import trade
from epymetheus import create_strategy
from epymetheus import Strategy


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
