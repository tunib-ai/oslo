import numpy as np


class WarmStartingMixin(object):
    def _warm_start_by_depth(self, target_hidden_layers):
        for policy in self.get_layer_policies():
            new_layer_indices = np.rint(
                np.arange(target_hidden_layers)
                / target_hidden_layers
                * self.config.num_hidden_layers
            ).astype(np.long)

            block_layers = policy.block_layers(self, self.config)
            new_blocks = block_layers.__class__()

            for new_layer_idx, current_layer_idx in enumerate(new_layer_indices):
                new_blocks[new_layer_idx] = current_layer_idx

            policy.set_block_layers(self, self.config, new_blocks)
            self.config.num_hidden_layers = target_hidden_layers

    def _warm_start_by_width(self, target_hidden_size):
        ...

    def warm_start(self, target_hidden_size, target_hidden_layers):
        self._warm_start_by_depth(target_hidden_layers=target_hidden_layers)
        self._warm_start_by_width(target_hidden_size=target_hidden_size)
        return self
