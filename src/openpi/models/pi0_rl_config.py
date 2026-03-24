import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0_rl import Pi0RL


@dataclasses.dataclass(frozen=True)
class Pi0RLConfig(pi0_config.Pi0Config):
    """Config for Pi0 with RL Token encoder-decoder (Stage 1 of RLT).

    The encoder-decoder operates at the VLM backbone's embedding dimension
    (determined by paligemma_variant). Default is 2048 for gemma_2b.
    """

    rl_num_layers: int = 2
    rl_num_heads: int = 8
    rl_mlp_dim: int = 8192
    # alpha in L_total = L_ro + alpha * L_vla.
    # Set to 0 to only train the encoder-decoder (VLA frozen w.r.t. all losses).
    rl_vla_loss_weight: float = 1.0

    @property
    def rl_embedding_dim(self) -> int:
        return _gemma.get_config(self.paligemma_variant).width

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RL":
        from openpi.models.pi0_rl import Pi0RL

        return Pi0RL(self, rngs=nnx.Rngs(rng))

    def get_rl_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze filter for RLT Stage 1 training.

        When rl_vla_loss_weight > 0, both VLA and encoder-decoder are trainable
        (stop_gradient prevents L_ro from affecting VLA params).

        When rl_vla_loss_weight == 0, freeze all VLA params so only the
        encoder-decoder trains.
        """
        if self.rl_vla_loss_weight > 0:
            return nnx.Nothing
        # Freeze everything except rl_encoder and rl_decoder
        return nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*rl_encoder.*")),
            nnx.Not(nnx_utils.PathRegex(".*rl_decoder.*")),
        )
