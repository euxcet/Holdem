import os
from ray.rllib.policy.policy import PolicySpec
from .random_heuristic import RandomHeuristic
from .random_range_heuristic import RandomRangeHeuristic
from .leduc.policy import LeducCFRHeuristic
from .leduc.range_policy import RangeLeducCFRHeuristic
from .kuhn.range_policy import RangeKuhnCFRHeuristic
from .hunl.policy import TfHeuristic
from ..config.train_config import TrainConfig

def get_policies(policies: list[str], files: TrainConfig.TrainPolicyConfig) -> dict:
    result = {}
    for policy in policies:
        if policy == 'random':
            result[policy] = PolicySpec(policy_class=RandomHeuristic)
        elif policy == 'random_range':
            result[policy] = PolicySpec(policy_class=RandomRangeHeuristic)
        elif policy == 'leduc_cfr':
            result[policy] = PolicySpec(policy_class=LeducCFRHeuristic,
                                        config={"nash": os.path.join(os.getcwd(), files.leduc_nash)})
        elif policy == 'range_leduc_cfr':
            result[policy] = PolicySpec(policy_class=RangeLeducCFRHeuristic,
                                        config={"nash": os.path.join(os.getcwd(), files.leduc_nash)})
        elif policy == 'range_leduc_cfr':
            result[policy] = PolicySpec(policy_class=RangeLeducCFRHeuristic,
                                        config={"nash": os.path.join(os.getcwd(), files.leduc_nash)})
        elif policy == 'range_kuhn_cfr':
            result[policy] = PolicySpec(policy_class=RangeKuhnCFRHeuristic,
                                        config={"nash": os.path.join(os.getcwd(), files.kuhn_nash)})
        elif policy == 'alphaholdem':
            result[policy] = PolicySpec(policy_class=TfHeuristic)
        else:
            raise Exception(f"{policy} does not exist.")
    return result