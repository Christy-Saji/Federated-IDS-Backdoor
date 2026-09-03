"""Back-compat shim. Phase 0 imports ``from flids.model import MLP``.

Canonical location is now ``flids.models.mlp``.
"""

from flids.models.mlp import LAYER_SIZES, MLP

__all__ = ["MLP", "LAYER_SIZES"]
