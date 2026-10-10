# Provenance of the released results

Every arm under `results/<agent>/<benchmark>/<regime>/` was produced by
`python -m vlnsafe run` with this repository's harness, **except three ETPNav arms**:

| file | origin | episodes |
|---|---|---|
| `results/etpnav/r2r/block/episodes.csv` | earlier full-split run of the same ETPNav checkpoint with the pre-release version of this harness | 1839 |
| `results/etpnav/rxr/block/episodes.csv` | same | 1223 |
| `results/etpnav/rxr/slide/episodes.csv` | same | 1223 |

These rows were migrated column-for-column to `docs/schema.md`: the pre-release
displacement-proxy count became `proxy_collision_count` and its engine-flag count became
`collision_count`. `scripts/check_equivalence.py` is the gate that compared a fresh
`vlnsafe run` arm against the migrated rows before they were accepted. Their
`manifest.json` records the original file name in `command`.
