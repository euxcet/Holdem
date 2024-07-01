from __future__ import annotations

import time
from .logger import log

class Counter():
    def __init__(
        self,
        execute_interval: int = 10,
        print_interval: int = 100
    ) -> None:
        self.start_time = time.time()
        self.counter = 0
        self.execute_interval = execute_interval
        self.print_interval = print_interval

    def reset(self) -> Counter:
        self.start_time = time.time()
        self.counter = 0
        return self

    def count(
        self,
        print_dict: dict = dict(),
        enable_print: bool = False,
        print_fps: bool = False,
    ) -> bool:
        self.counter += 1
        if enable_print and self.counter % self.print_interval == 0:
            current_time = time.time()
            if print_fps:
                print_dict['FPS'] = self.print_interval / (current_time - self.start_time)
            log.info('  '.join(f'{k}: {v}' for k, v in print_dict.items()))
            self.start_time = current_time
        return self.counter % self.execute_interval == 0