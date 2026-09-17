"""A local, read-mostly dashboard over results/ plus a live federated demo.

    python -m flids.dashboard

Exists for the viva: a judge can watch the federation train, watch four
malicious clients drag a backdoor into the global model, and then poke a
*trained* model until it misclassifies a real CIC-IDS2017 flow on demand, and
put all five defenses side by side on the recorded runs.

Two deliberate constraints:

* **No new dependencies.** The server is ``http.server`` from the stdlib and the
  page is hand-written HTML/CSS/JS. ``requirements.txt`` is pinned because Gate
  G1 wants byte-identical runs on three machines; a dashboard is not a good
  enough reason to put a web framework between the team and that guarantee.
* **It never writes to ``results/``.** Live simulations are held in memory and
  discarded. Everything reproducible goes through ``flids.runner``, which
  assigns a run_id and refuses to overwrite. A demo must not be able to
  manufacture a result.
"""

import os

# Before numpy loads: OpenBLAS oversubscribes on matrices this small, and a
# single thread makes a live round about twice as fast. Live simulations are
# never recorded, so bit-identity across thread counts does not apply here; an
# explicit OPENBLAS_NUM_THREADS in the environment still wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from .engine import DemoEngine  # noqa: E402

__all__ = ["DemoEngine"]
