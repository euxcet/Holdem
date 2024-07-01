from __future__ import annotations

from enum import Enum

class Street(Enum):
    Preflop = 0
    Flop = 1
    Turn = 2
    River = 3
    Showdown = 4

    def from_str(s: str) -> Street:
        s = s.strip().lower()
        if s == 'preflop':
            return Street.Preflop
        elif s == 'flop':
            return Street.Flop
        elif s == 'turn':
            return Street.Turn
        elif s == 'river':
            return Street.River
        else:
            return Street.Showdown

    def next(self) -> Street | None:
        if (self == Street.Preflop): return Street.Flop
        elif (self == Street.Flop): return Street.Turn
        elif (self == Street.Turn): return Street.River
        elif (self == Street.River): return Street.Showdown
        else: return Street.Showdown
