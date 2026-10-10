"""VLN-SAFE harness.

What: runs one frozen VLN-CE checkpoint through Habitat for a chosen benchmark and
collision regime, and writes the two-CSV log contract in docs/schema.md.

Who reads it: scripts/launch_arm.sh (one shard), analysis/ (the logs afterwards).

Must not import at package level: habitat, torch. Importing `vlnsafe` has to stay cheap
enough for `python -m vlnsafe describe` and for the pure tests, which run without habitat
installed. Submodules that need habitat import it inside their functions.

Python 3.6 compatible throughout this package (habitat-sim 0.1.7 pins it): no dataclasses,
no walrus, no `from __future__ import annotations`.
"""

__version__ = "0.1.0"
