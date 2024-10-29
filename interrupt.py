import numpy as np


class Interrupt(object):
    def __init__(self):
        self.states = [1.011, 1.012, 1.013, 1.014]
        self.transition_matrix = np.ones((len(self.states), len(self.states))) * 0.25
        self.recovery_time = 0

    def reset(self):
        self.state_idx = np.random.choice(list(range(len(self.states))))
        self.recovery_time = 0
        return self.state

    def step(self, power):
        if self.recovery_time > 0:
            self.recovery_time -= 1
        else:
            self.state_idx = np.random.choice(
                list(range(len(self.states))), p=self.transition_matrix[self.state_idx]
            )
            self.sample_interrupt(power)

        return self.is_interrupt

    def sample_interrupt(self, power):
        interrupt_prob = (self.state**power - 1) / 100
        print("interrupt_prob:", interrupt_prob)

        if np.random.rand() < interrupt_prob:
            self.recovery_time = np.random.randint(4, 7)

        return self.is_interrupt

    @property
    def state(self):
        return self.states[self.state_idx]

    @property
    def is_interrupt(self):
        return self.recovery_time != 0


if __name__ == "__main__":
    interrupt = Interrupt()
    import time

    interrupt.reset()

    for _ in range(200):
        time.sleep(1)
        power = np.random.randint(1, 10) * 30
        interrupt.step(power)
        print("power:", power)
        print("interrupt:", interrupt.is_interrupt)
        print("state:", interrupt.state)
        print("recovery_time:", interrupt.recovery_time)
        print("===")
