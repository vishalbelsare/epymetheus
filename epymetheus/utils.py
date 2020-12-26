from .strategy import Strategy


class CombinedStrategy(Strategy):
    """
    Combine multiple strategies.

    >>> import epymetheus as ep

    >>> logic1 = lambda universe: [ep.trade("A")]
    >>> logic2 = lambda universe: [ep.trade("B"), ep.trade("C")]
    >>> strategy1 = ep.create_strategy(logic1)
    >>> strategy2 = ep.create_strategy(logic2)
    >>> combined_strategy = CombinedStrategy(strategy1, strategy2)
    >>> universe = ...
    >>> combined_strategy(universe)
    [trade(['A'], lot=[1.]), trade(['B'], lot=[1.]), trade(['C'], lot=[1.])]
    """

    def __init__(self, *strategies):
        super().__init__()

        self.strategies = strategies

    def __repr__(self):
        return f"{self.__class__.__name__}{self.strategies}"

    def __getitem__(self, i):
        return self.strategies[i]

    def append(self, strategy):
        self.strategies.append(strategy)

    def logic(self, universe):
        for strategy in self.strategies:
            for trade in strategy(universe):
                yield trade
