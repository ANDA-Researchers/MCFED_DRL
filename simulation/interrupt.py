import numpy as np


class Interrupt(object):
    def __init__(self, enable=True, probs=1.011):
        self.state = probs
        self.recovery_time = 0
        self.enable = enable

    def reset(self):
        self.recovery_time = 0
        return self.state

    def step(self, power):
        if self.recovery_time > 0:
            self.recovery_time -= 1
        else:
            self.sample_interrupt(power)

        return self.is_interrupt

    def sample_interrupt(self, power):
        interrupt_prob = (self.state**power - 1) / 100
        rate = np.random.rand()
        print(rate, interrupt_prob)
        if rate < interrupt_prob:
            self.recovery_time = 2

        return self.is_interrupt

    @property
    def is_interrupt(self):
        if not self.enable:
            return False
        return self.recovery_time != 0


if __name__ == "__main__":
    interrupt = Interrupt()
    import time

    interrupt.reset()

    for _ in range(200):
        time.sleep(1)
        power = np.random.randint(8, 10) * 30
        interrupt.step(power)
        print("power:", power)
        print("interrupt:", interrupt.is_interrupt)
        print("state:", interrupt.state)
        print("recovery_time:", interrupt.recovery_time)
        print("===")
