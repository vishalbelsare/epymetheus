import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn


def print_as_comment(obj):
    print("\n".join(f"# {line}" for line in str(obj).splitlines()))


if __name__ == "__main__":
    sys.path.append("../..")
    seaborn.set_style("whitegrid")

    # ---

    import pandas as pd
    import epymetheus as ep

    def dumb_strategy(universe: pd.DataFrame, profit_take, stop_loss):
        # I get $100 allowance on the first business day of each month
        allowance = 100

        trades = []
        for date in pd.date_range(universe.index[0], universe.index[-1], freq="BMS"):
            cheapest_stock = universe.loc[date].idxmin()

            # Find the maximum number of shares that I can buy with my allowance
            n_shares = allowance // universe.at[date, cheapest_stock]

            trade = n_shares * ep.trade(
                cheapest_stock,
                open_bar=date,
                take=profit_take,
                stop=stop_loss,
            )
            trades.append(trade)

        return trades

    # ---

    my_strategy = ep.create_strategy(dumb_strategy, profit_take=20.0, stop_loss=-10.0)

    # ---

    from epymetheus.datasets import fetch_usstocks

    universe = fetch_usstocks()
    print(">>> universe.head()")
    print_as_comment(universe.head())

    print(">>> my_strategy.run(universe)")
    my_strategy.run(universe)

    # ---

    history = my_strategy.history.to_dataframe()
    history.head()
    print(">>> history.head()")
    print_as_comment(history.head())

    print(sum(history.pnl))

    # ---

    series_wealth = my_strategy.wealth()

    print(">>> series_wealth.head()")
    print_as_comment(series_wealth.head())

    plt.figure(figsize=(16, 4))
    plt.plot(series_wealth, linewidth=1)
    plt.xlabel("date")
    plt.ylabel("wealth [USD]")
    plt.title("Wealth")
    plt.savefig("wealth.png", bbox_inches="tight", pad_inches=0.1)

    # ---

    from epymetheus.metrics import Drawdown
    from epymetheus.metrics import Exposure
    from epymetheus.metrics import MaxDrawdown
    from epymetheus.metrics import SharpeRatio

    drawdown = my_strategy.score(Drawdown())
    max_drawdown = my_strategy.score(MaxDrawdown())
    net_exposure = my_strategy.score(Exposure(net=True))
    abs_exposure = my_strategy.score(Exposure(net=False))
    sharpe_ratio = my_strategy.score(SharpeRatio())

    plt.figure(figsize=(16, 4))
    plt.plot(pd.Series(drawdown, index=universe.index), linewidth=1)
    plt.xlabel("date")
    plt.ylabel("drawdown [USD]")
    plt.title("Drawdown")
    plt.savefig("drawdown.png", bbox_inches="tight", pad_inches=0.1)

    plt.figure(figsize=(16, 4))
    plt.plot(pd.Series(net_exposure, index=universe.index), linewidth=1)
    plt.xlabel("date")
    plt.ylabel("net exposure [USD]")
    plt.title("Net exposure")
    plt.savefig("net_exposure.png", bbox_inches="tight", pad_inches=0.1)

    # ---

    plt.figure(figsize=(16, 4))
    plt.hist(my_strategy.history.pnl, bins=100)
    plt.axvline(0, ls="--", color="k")
    plt.xlabel("profit and loss")
    plt.ylabel("number of trades")
    plt.title("Profit-loss distribution")
    plt.savefig("pnl.png", bbox_inches="tight", pad_inches=0.1)
