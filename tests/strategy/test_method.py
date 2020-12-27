import pandas as pd
import pytest

import epymetheus as ep
from epymetheus import Strategy
from epymetheus.benchmarks import DeterminedStrategy
from epymetheus.datasets import make_randomwalk
from epymetheus.exceptions import NoTradeError


class HardCodedStrategy(Strategy):
    """
    Yield hard-coded trades.
    """

    trade0 = ep.trade(asset="A0", lot=1.0, entry="B0", exit="B1", take=2.0, stop=-2.0)
    trade1 = ep.trade(asset="A1", lot=1.1, entry="B2", exit="B3", take=2.1, stop=-2.1)

    def logic(self, universe):
        yield self.trade0
        yield self.trade1


class NoTradeStrategy(Strategy):
    """
    Yields no strategy. Used in `TestRun.test_no_trade_error`.
    """

    def logic(self, universe):
        pass


class TestRun:
    """
    Test `Strategy.run()`.
    """

    params_verbose = [True, False]

    universe = prices = pd.DataFrame(
        {f"A{i}": range(10) for i in range(10)}, index=[f"B{i}" for i in range(10)]
    )

    @pytest.mark.parametrize("verbose", params_verbose)
    def test_generate(self, verbose):
        """
        Test if `run()` generates correct trades for a hard-coded strategy.
        """
        strategy = HardCodedStrategy().run(self.universe, verbose=verbose)
        expected = [strategy.trade0, strategy.trade1]
        assert strategy.trades == expected

    @pytest.mark.parametrize("verbose", params_verbose)
    def test_close(self, verbose):
        """
        Notes
        -----
        Correctness of execution itself is tested for Trade class.
        """
        strategy = HardCodedStrategy().run(self.universe, verbose=verbose)
        expected = [
            t.execute(self.universe).close
            for t in (strategy.trade0, strategy.trade1)
        ]
        result = [trade.close for trade in strategy.trades]
        assert result == expected

    # @pytest.mark.parametrize("verbose", params_verbose)
    # def test_pnl(self, verbose):
    #     """
    #     Notes
    #     -----
    #     Correctness of execution itself is tested for Trade class.
    #     """
    #     strategy = HardCodedStrategy().run(self.universe, verbose=verbose)
    #     result = [trade.pnl for trade in strategy.trades]
    #     expected = [trade.execute(self.universe).pnl for trade in strategy.trades]
    #     assert result == expected

    @pytest.mark.parametrize("verbose", params_verbose)
    def test_no_trade_error(self, verbose):
        strategy = NoTradeStrategy()
        with pytest.raises(NoTradeError):
            strategy.run(make_randomwalk(), verbose=verbose)


class TestCompile:
    pass  # TODO


class TestEvaluate:
    pass  # TODO
