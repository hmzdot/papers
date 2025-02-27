import numpy as np
from rbm import RBM


class DBN:
    """Deep Belief Network"""

    layers: list[RBM]
    """RBM layers"""

    def __init__(self, layer_sizes: list[int], batch_size=32) -> None:
        num_layers = len(layer_sizes) - 1
        self.layers = [
            RBM(
                num_visible=layer_sizes[i],
                num_hidden=layer_sizes[i + 1],
                batch_size=batch_size,
            )
            for i in range(num_layers)
        ]

    def pretrain(self, data: np.ndarray, epochs=10, lr=0.1):
        """Train each RBM in a greedy layer-wise manner"""
        for i, rbm in enumerate(self.rbms):
            print(
                "Training RBM %d/%d (%d → %d)"
                % (i + 1, self.num_layers, rbm.num_visible, rbm.num_hidden)
            )
            rbm.train(data, epochs=epochs, lr=lr)
            data = rbm.sample_hidden(data)
