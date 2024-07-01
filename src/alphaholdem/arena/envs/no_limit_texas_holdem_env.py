from ...poker.component.street import Street
from ...poker.no_limit_texas_holdem_env import NoLimitTexasHoldemEnv

def create_no_limit_holdem_env() -> NoLimitTexasHoldemEnv:
    return NoLimitTexasHoldemEnv(
        num_players=2,
        initial_chips=200,
        showdown_street=Street.Showdown,
        circular_train=False,
        payoff_max=200,
    )