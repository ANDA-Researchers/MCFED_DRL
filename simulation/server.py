import numpy as np
from .interrupt import Interrupt


class RSU:
    def __init__(self, position: tuple, capacity: int, model, num_items) -> None:
        self.position = position
        self.capacity = capacity
        self.cache = np.random.randint(0, num_items, capacity)
        self.model = model
        self.cluster = None
        self.interrupt = Interrupt()
        self.interrupt.reset()

    def had(self, data: int) -> bool:
        return data in self.cache

    def is_interrupt(self):
        return self.interrupt.is_interrupt

    def step(self, power):
        self.interrupt.step(power)
        return self.interrupt.is_interrupt


class BS:
    def __init__(self, position: tuple) -> None:
        self.position = position

    def had(self, data: int) -> bool:
        return True
