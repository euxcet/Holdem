from rich.table import Table
from enum import Enum
from .street import Street

class ActionType(Enum):
    Fold = 0
    Check = 1
    Call = 2
    Raise = 3
    All_in = 4
    
    def default_legal() -> list[bool]:
        return [True, True, True, True, True]
    
    # All-in or fold
    def aof_legal() -> list[bool]:
        return [True, False, False, False, True]

class Action():
    def __init__(
        self,
        type: ActionType,
        player: int,
        player_name: str,
        street: Street,
        raise_chip: int = 0,
        raise_pot: float = 0,
        raise_to: int = 0,
        all_in_chip: int = 0,
        all_in_type: ActionType = ActionType.Call,
    ) -> None:
        self.type = type
        self.player = player
        self.player_name = player_name
        self.street = street
        self.raise_chip = raise_chip
        self.raise_pot = raise_pot
        self.raise_to = raise_to
        self.all_in_chip = all_in_chip
        self.all_in_type = all_in_type

    def __str__(self) -> str:
        return f"[Action] player={self.player_name} street={self.street} action={self.type} \n" + \
               f"         raise_chip={self.raise_chip / 2}BB \t raise_pot={self.raise_pot * 100}% \t " + \
               f"raise_to={self.raise_to / 2}BB" + \
                "\n"
    
    def __repr__(self) -> str:
        return self.__str__()