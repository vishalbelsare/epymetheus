import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn

import optuna


seaborn.set_style("whitegrid")


def print_as_comment(obj):
    print("\n".join(f"# {line}" for line in str(obj).splitlines()))


if __name__ == "__main__":
    sys.path.append("../..")

    import epymetheus as ep

    def dumb_strategy(universe: pd.DataFrame, profit_take=20, stop_loss=-10):
        """
        Buy the cheapest stock every month with my allowance.

        Parameters
        ----------
        - profit_take : float, default None
            Threshold (in unit of USD) to make profit-take order.
        - stop_loss : float, default None
            Threshold (in unit of USD) to make stop-loss order.

        Yields
        ------
        trade : ep.trade
            Trade object.
        """
        # I get allowance on the first business day of each month
        allowance = 100
        allowance_dates = pd.date_range(
            universe.index[0], universe.index[-1], freq="BMS"
        )

        trades = []
        for date in allowance_dates:
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

    from epymetheus.datasets import fetch_usstocks
    from epymetheus.metrics import FinalWealth

    universe = fetch_usstocks()

    def objective(trial):
        profit_take = trial.suggest_int("profit_take", 10, 100)
        stop_loss = trial.suggest_int("stop_loss", -100, -10)
        my_strategy = ep.create_strategy(
            dumb_strategy,
            profit_take=profit_take,
            stop_loss=stop_loss,
        )
        my_strategy.run(universe, verbose=False)

        return my_strategy.score(FinalWealth())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print(">>> study.best_params")
    print_as_comment(study.best_params)
