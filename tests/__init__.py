"""Unit tests. Stdlib ``unittest`` on purpose - no new dependency.

    python -m unittest discover -s tests -t .

``requirements.txt`` is pinned because Gate G1 wants byte-identical runs on
three machines, so the test suite does not get to add pytest to it. Everything
here runs on the same interpreter the campaign runs on.

These tests pin the things that have silently broken before. Every gate in
``scripts/gates/`` checks the pipeline end to end, which is why the two worst
bugs in this project's history - ``MLP.set_params`` returning views that the
optimiser then mutated in place, and Activation Clustering being handed the
clean split instead of the poisoned rows - both survived until somebody
eyeballed a result that looked wrong. A gate says "the number exists"; these
say "the number means what it says".
"""
