import pytest

from epymetheus import Strategy, History
from epymetheus.exceptions import NotRunError


class SampleStragegy(Strategy):
    """
    A sample strategy.
    """

    def __init__(self):
        pass

    def logic(self, universe):
        pass


# --------------------------------------------------------------------------------


def test_notrunerror():
    strategy = SampleStragegy()

    with pytest.raises(NotRunError):
        History(strategy)
