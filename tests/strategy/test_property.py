import pytest  # noqa: F401

import pandas as pd

from epymetheus import Strategy, Trade


class SampleStrategy(Strategy):
    """
    This is my favorite strategy.

    Everybody loves this strategy.
    """

    def __init__(self, param0=None, param1=None):
        self.param0 = param0
        self.param1 = param1

    def logic(self, univers):
        yield Trade(asset=["A0", "A1"])
        yield Trade(asset=["A2", "A3"])





# class TestIsRun:
#     """
#     Test `Strategy.is_run`.
#     """

#     universe = pd.DataFrame({f"A{i}": range(10) for i in range(4)})

#     def test_value(self):
#         strategy = SampleStrategy()
#         assert strategy.is_run == False
#         strategy.run(self.universe)
#         assert strategy.is_run == True


# class TestParams:
#     """
#     Test `Strategy.params`.
#     """

#     def test_value(self):
#         strategy = SampleStrategy(param0=0.0, param1=1.0)
#         assert strategy.params == {"param0": 0.0, "param1": 1.0}


class TestHistory:
    def test_value(self):
        # TODO: move from tests/history
        pass


class TestWealth:
    def test_value(self):
        # TODO: move from tests/history
        pass
