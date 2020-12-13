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

    def dumb_strategy(universe, allowance=100.0, take=None, stop=None):
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
                n_shares * ep.trade(cheapest_stock, open_bar=date, take=take, stop=stop)
            )

    from epymetheus.datasets import fetch_usstocks
    from epymetheus.metrics import FinalWealth

    universe = fetch_usstocks()
    print(">>> universe.prices")
    print_as_comment(universe.prices)

    def objective(trial):
        take = trial.suggest_int("take", 10, 100)
        stop = trial.suggest_int("stop", -100, -10)
        my_strategy = ep.create_strategy(dumb_strategy, take=take, stop=stop)
        my_strategy.run(universe, verbose=False)

        return my_strategy.score(FinalWealth())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print(">>> study.best_params")
    print_as_comment(study.best_params)
