# Epymetheus: Python Library for Multi-asset Backtesting

[![python versions](https://img.shields.io/pypi/pyversions/epymetheus.svg)](https://pypi.org/project/epymetheus/)
[![version](https://img.shields.io/pypi/v/epymetheus.svg)](https://pypi.org/project/epymetheus/)
[![CI](https://github.com/simaki/epymetheus/workflows/CI/badge.svg)](https://github.com/simaki/epymetheus/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/simaki/epymetheus/branch/master/graph/badge.svg)](https://codecov.io/gh/simaki/epymetheus)
[![dl](https://img.shields.io/pypi/dm/epymetheus)](https://pypi.org/project/epymetheus/)
[![LICENSE](https://img.shields.io/github/license/simaki/epymetheus)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![wealth](examples/readme/wealth.png)

## Introduction

Epymetheus is a Python library for multi-asset backtesting.
It provides an end-to-end framework that lets analysts build and try out their trade strategy right away.

### Features

1. **Simple and Intuitive API**: The API is simply and intuitively designed so that you can focus on your idea. Trade strategy is readily coded as a usual function, and then you can `run()` and `score()` it right away.
2. **Seamless connection to [Pandas](https://github.com/pandas-dev/pandas)**: You can just use `pandas.DataFrame` of historical prices as the target of backtesting. Backtesting results can be quickly converted to `pandas.DataFrame` so that you can view, analyze and plot results by the familiar Pandas methods.
3. **Extensibility with Other Frameworks**: Epymetheus only provides a framework.  Strategy can be readily built with other libraries for machine learning, econometrics, technical indicators, hyper-parameter optimization framework and so forth.
4. **Efficient Computation**: Backtesting engine is boosted by NumPy.  You can give your own idea a quick try.
5. **Full Test Coverage**: Epymetheus is thoroughly tested with 100% test coverage for multiple Python versions.

### Integrations

- **Machine Learning**: [scikit-learn](https://github.com/scikit-learn/scikit-learn), [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), etc.
- **Econometrics**: [statsmodels](https://github.com/statsmodels/statsmodels), etc.
- **Technical Indicators**: [TA-Lib](https://github.com/mrjbq7/ta-lib), etc.
- **Hyperparameter Optimization**: [optuna](https://github.com/optuna/optuna). Example follows.

## Installation

```sh
$ pip install epymetheus
```

## How to use

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simaki/epymetheus/blob/master/examples/readme/readme.ipynb)

### Create strategy

Let's construct your own strategy.  This strategy will:

* buy the cheapest stock with your monthly allowance $100.0.
* take profit if it exceeds $10.0 and make stop-loss order if the loss exceeds -$10.0.

```python
import epymetheus as ep


def dumb_strategy(universe, profit_take=10.0, stop_loss=-10.0):
    # I get allowance on the first business day of each month
    allowance = 100.0
    allowance_dates = pd.date_range(universe.prices.index[0], universe.prices.index[-1], freq="BMS")
    
    trades = []
    for date in allowance_dates:
        cheapest_stock = universe.prices.loc[date].idxmin()

        # Find the maximum number of shares that I can buy with my allowance
        n_shares = allowance // universe.prices.at[date, cheapest_stock]

        trade = n_shares * ep.trade(
            cheapest_stock,
            open_bar=date,
            take=profit_take,
            stop=stop_loss,
        )
        trades.append(trade)
        
    return trades
```

Here,

* The first parameter `universe` is mandatory. It means a set of assets that you will trade (US stocks, cryptocurrencies, etc).
* The following parameters, `profit_take` and `stop_loss` parametrize your strategy.

You can create your strategy as:

```python
my_strategy = ep.create_strategy(dumb_strategy, profit_take=20.0, stop_loss=-10.0)
```

### Run strategy

Now your strategy is readily backtested with any `Universe`.

```python
from epymetheus.datasets import fetch_usstocks

universe = fetch_usstocks(n_assets=10)
universe.prices
#                  AAPL        MSFT         AMZN   BRK-A         JPM         JNJ         WMT        BAC          PG        XOM
# 2000-01-01   0.785456   37.162327    76.125000   56100   27.773939   27.289129   46.962898  14.527933   31.304089  21.492596
# 2000-01-02   0.785456   37.162327    76.125000   56100   27.773939   27.289129   46.962898  14.527933   31.304089  21.492596
# 2000-01-03   0.855168   37.102634    89.375000   54800   26.053429   26.978193   45.391777  14.021359   30.625511  20.892334
# 2000-01-04   0.783068   35.849308    81.937500   52000   25.481777   25.990519   43.693306  13.189125   30.036228  20.492161
# 2000-01-05   0.794528   36.227283    69.750000   53200   25.324482   26.264877   42.801613  13.333860   29.464787  21.609318

my_strategy.run(universe)
# Yield 240 trades: trade(['BAC'], lot=[3.], open_bar=2019-12-02 00:00:00) ... Done. (Runtime : 0.12 sec)
# Execute 240 trades: trade(['BAC'], lot=[3.], open_bar=2019-12-02 00:00:00) ... Done. (Runtime : 0.03 sec)
# Done. (Runtime : 0.15 sec)
```

### Trade history and wealth

Trade history can be viewed as:

```python
df_history = my_strategy.history.to_dataframe()
df_history.head()
#           trade_id asset    lot   open_bar  close_bar shut_bar  take  stop        pnl
# order_id                                                                             
# 0                0  AAPL  116.0 2000-01-03 2000-01-06     None  20.0 -10.0 -15.010098
# 1                1  AAPL  130.0 2000-02-01 2000-03-01     None  20.0 -10.0  29.856866
# 2                2  AAPL  100.0 2000-03-01 2000-03-14     None  20.0 -10.0 -12.271219
# 3                3  AAPL   98.0 2000-04-03 2000-04-11     None  20.0 -10.0 -10.388053
# 4                4  AAPL  105.0 2000-05-01 2000-05-04     None  20.0 -10.0 -10.929523
```

The time-series of wealth can be viewed as:

```python
df_wealth = my_strategy.wealth.to_dataframe()
df_wealth.head()
#               wealth
# bars
# 2000-01-01  0.000000
# 2000-01-02  0.000000
# 2000-01-03  0.000000
# 2000-01-04 -8.363557
# 2000-01-05 -7.034265
```

![wealth](examples/readme/wealth.png)

### Scores

You can also quickly `score` the metrics of the perfornance.

```python
from epymetheus.metrics import Drawdown
from epymetheus.metrics import MaxDrawdown
from epymetheus.metrics import SharpeRatio

drawdown = my_strategy.score(Drawdown())
max_drawdown = my_strategy.score(MaxDrawdown())
net_exposure = my_strategy.score(Exposure(net=True))
abs_exposure = my_strategy.score(Exposure(net=False))
sharpe_ratio = my_strategy.score(SharpeRatio())
```

![drawdown](examples/readme/drawdown.png)
![net_exposure](examples/readme/net_exposure.png)

## More examples

### Optimization

You may optimize the parameter of your strategy using, for example, optuna.
(Remember that optimization for backtesting is dangerous.)

```python
import optuna


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

study.best_params
# {'profit_take': 100, 'stop_loss': -83}
```

### Pair trading

Trade can include multiple stocks.
Profit-take and/or stop-loss will be executed when the total profit/loss exceeds the thresholds.

```python
def pair_trading(universe, param_1, ...):
    ...
    # Buy 1 share of "BULLISH_STOCK" and sell 2 share of "BEARISH_STOCK".
    yield [1.0, -2.0] * ep.trade(["BULLISH_STOCK", "BEARISH_STOCK"], stop=-100.0)
```
