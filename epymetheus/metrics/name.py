from .metrics import avg_lose
from .metrics import avg_pnl
from .metrics import avg_win
from .metrics import final_wealth
from .metrics import num_lose
from .metrics import num_win
from .metrics import rate_lose
from .metrics import rate_win


def metric_from_name(name: str):
    """
    Return metrics from name.
    """
    metrics = (
        final_wealth,
        num_win,
        num_lose,
        rate_win,
        rate_lose,
        avg_win,
        avg_lose,
        avg_pnl,
    )
    return {m.__name__: m for m in metrics}[name]
