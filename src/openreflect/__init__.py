"""OpenReflect: a fully specified recipe for training long-horizon self-improving agents.

Builds on AREX-2 (Qian et al., 2026, arXiv:2609.38288). Components:

- ``openreflect.envs``       environment spec, normalized scores, headroom filter,
                             decontamination, isolated scorer, process audit, synthesis
- ``openreflect.scaffold``   fixed tool set, round ledger, Round Ledger Compaction (RLC),
                             agent loop
- ``openreflect.palm``       Progress-Aware Loss Masking
- ``openreflect.selection``  trajectory acceptance rules
- ``openreflect.train``      offline RLC windowing and weighted SFT
- ``openreflect.metrics``    process metrics (T*, mean gain, AUC, recovery, wasted turns)
"""

__version__ = "0.1.0"
