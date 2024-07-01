from ray.rllib.policy.policy import PolicySpec
from .random_heuristic import RandomHeuristic
from .leduc.policy import LeducCFRHeuristic
from .hunl.policy import TfHeuristic

def get_policies(policies: list[str]) -> dict:
    result = {}
    for policy in policies:
        if policy == 'random':
            result[policy] = PolicySpec(policy_class=RandomHeuristic)
        elif policy == 'leduc_cfr':
            result[policy] = PolicySpec(policy_class=LeducCFRHeuristic)
        elif policy == 'alphaholdem':
            result[policy] = PolicySpec(policy_class=TfHeuristic)
        else:
            raise Exception(f"{policy} does not exist.")
    return result