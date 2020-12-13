import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn


seaborn.set_style("whitegrid")


def print_as_comment(obj):
    print("\n".join(f"# {line}" for line in str(obj).splitlines()))


if __name__ == "__main__":
    sys.path.append("../..")

    # ---

    import epymetheus as ep

    def dumb_strategy(universe, allowance):
        """
        Buy the cheapest stock every month with my allowance.
        """
        # I get allowance on the first business day of each month
        allowance_dates = pd.date_range(
            universe.prices.index[0], universe.prices.index[-1], freq="BMS"
        )

        for date in allowance_dates:
            # Find the cheapest stock
            cheapest_stock = universe.prices.loc[date].idxmin()
            # Find the maximum number of shares that I can buy with my allowance
            n_shares = allowance // universe.prices.at[date, cheapest_stock]
            # Trade!
            yield (
                n_shares
                * ep.trade(cheapest_stock, open_bar=date, take=20.0, stop=-10.0)
            )

    # ---

    my_strategy = ep.create_strategy(dumb_strategy, allowance=100.0)

    # ---

    from epymetheus.datasets import fetch_usstocks

    universe = fetch_usstocks()
    print(">>> universe.prices")
    print_as_comment(universe.prices)

    my_strategy.run(universe)

    # ---

    df_history = my_strategy.history.to_dataframe()
    df_history.head()
    print(">>> df_history.head()")
    print_as_comment(df_history.head())

    # ---

    df_wealth = my_strategy.wealth.to_dataframe()

    print(">>> my_strategy.wealth.to_dataframe().head()")
    print_as_comment(my_strategy.wealth.to_dataframe().head())

    plt.figure(figsize=(16, 4))
    plt.plot(df_wealth, linewidth=1)
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
    plt.plot(pd.Series(drawdown, index=universe.prices.index), linewidth=1)
    plt.xlabel("date")
    plt.ylabel("drawdown [USD]")
    plt.title("Drawdown")
    plt.savefig("drawdown.png", bbox_inches="tight", pad_inches=0.1)

    plt.figure(figsize=(16, 4))
    plt.plot(pd.Series(net_exposure, index=universe.prices.index), linewidth=1)
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
