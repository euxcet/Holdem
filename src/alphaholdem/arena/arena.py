from abc import ABC, abstractmethod
from .policy.policy import Policy

class Arena(ABC):
    def __init__(self):
        ...

    @property
    @abstractmethod
    def nash_policy(self) -> Policy:
        ...

    @abstractmethod
    def validate_policy(self, policy: Policy) -> None:
        ...

    @abstractmethod
    def policy_vs_policy(
        self,
        policy0: Policy,
        policy1: Policy,
        runs: int = 1024,
    ) -> tuple[float, float]:
        ...

    def policies_melee(
        self,
        policies: list[Policy],
        runs: int = 1024,
    ) -> list[float]:
        scores = [0] * len(policies)
        for i in range(len(policies)):
            for j in range(i + 1, len(policies)):
                mean, var = self.policy_vs_policy(
                    policy0=policies[i],
                    policy1=policies[j],
                    runs=runs
                )
                scores[i] += mean / (len(policies) - 1)
                scores[j] -= mean / (len(policies) - 1)
        return scores