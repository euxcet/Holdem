from ...poker.component.street import Street
from ...poker.limit_leduc_holdem_env import LimitLeducHoldemEnv

def create_limit_holdem_env() -> LimitLeducHoldemEnv:
    return LimitLeducHoldemEnv(
        num_players=2,
        initial_chips=100,
        showdown_street=Street.Turn,
        circular_train=True,
        payoff_max=15,
    )