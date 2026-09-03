"""flids - Federated IDS backdoor research package.

Layout (Phase 1, Task 1.4):
  data/    loaders.py  labels.py  partition.py           (M2 / M3)
  models/  mlp.py  tabtransformer.py  registry.py        (M2)
  fl/      server.py  client.py  aggregators/            (M1 / M3)
  attacks/ badnets.py                                    (M1)
  eval/    metrics.py                                    (M3)
  utils/   seeding.py
  runner.py                                              (M1, shared contract)

No torch dependency: models are pure numpy so Gate G1 (byte-identical runs on
three machines) does not fight a CUDA nondeterminism surface.

``flids.model`` and ``flids.backdoor`` are back-compat shims for Phase 0.
"""

__all__ = ["data", "models", "fl", "attacks", "eval", "utils", "runner"]
