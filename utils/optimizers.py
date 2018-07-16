import numpy as np

"""
introduce some useful optimizers.
AdaGrad, Momentum
"""


class AdaGrad:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.h = {}

    def update(self, weights, grads, zero_constant=1e-7):
        if zero_constant == 0:
            raise ZeroDivisionError("can not divide by zero.")
        for k, v in weights.items():
            self.h[k] = np.zeros_like(v)

        for key in weights.keys():
            self.h[k] += grads[k] * grads[k]
            weights[k] -= (
                self.learning_rate * grads[key] / (np.sqrt(self.h[key]) + zero_constant)
            )

