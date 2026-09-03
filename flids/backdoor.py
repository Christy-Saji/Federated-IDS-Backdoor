"""Back-compat shim. Phase 0 imports ``from flids.backdoor import ...``.

Canonical location is now ``flids.attacks.badnets``.
"""

from flids.attacks.badnets import (INBOUNDS_VALUE, TARGET_LABEL, TRIGGER_VALUE,
                                   evaluate_backdoor, evaluate_model,
                                   poison_split, stamp_trigger)

__all__ = ["stamp_trigger", "poison_split", "evaluate_model",
           "evaluate_backdoor", "TARGET_LABEL", "TRIGGER_VALUE",
           "INBOUNDS_VALUE"]
