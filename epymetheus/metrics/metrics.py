import numpy as np

from .. import ts


def _pnls(trades, universe):
    return np.array([t.final_pnl(universe) for t in trades])


def final_wealth(trades, universe):
    # maybe faster than wealth[-1]
    return sum(t.final_pnl(universe) for t in trades)


def num_win(trades, universe):
    pnls = _pnls(trades, universe)
    return np.sum(pnls > 0)


def num_lose(trades, universe):
    pnls = _pnls(trades, universe)
    return np.sum(pnls <= 0)


def rate_win(trades, universe):
    return num_win(trades, universe) / len(trades)


def rate_lose(trades, universe):
    return num_lose(trades, universe) / len(trades)


def avg_win(trades, universe):
    pnls = _pnls(trades, universe)
    return np.mean(pnls[pnls > 0])


def avg_lose(trades, universe):
    pnls = _pnls(trades, universe)
    return np.mean(pnls[pnls <= 0])


def avg_pnl(trades, universe):
    pnls = _pnls(trades, universe)
    return np.mean(pnls)


def max_drawdown(trades, universe):
    return np.min(ts.drawdown(trades, universe))


# def avg_return(trades, universe):
#     return ...


# def volatility(trades, universe):
#     wealth = ts.wealth(trades, universe)
#     return np.std(np.diff(wealth))


# def sharpe_ratio(trades, universe):
#     return ...
