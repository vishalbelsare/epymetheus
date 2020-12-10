import pytest

from epymetheus import Strategy


class StrategyWithoutLogic(Strategy):
    """
    Strategy without logic.
    """

    def __init__(self, param=None):
        self.param = param


# --------------------------------------------------------------------------------


class TestABC:
    def test_abc_abstract(self):
        """
        One cannot instantiate `Strategy` itself.
        """
        with pytest.raises(TypeError):
            Strategy()

    def test_abc_nologic(self):
        """
        One cannot instantiate strategy without logic.
        """
        with pytest.raises(TypeError):
            StrategyWithoutLogic()
