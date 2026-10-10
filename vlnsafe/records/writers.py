"""Streaming CSV writers plus the run manifest.

What: EpisodeWriter and StepWriter open a file, write the schema header once, and flush
after every episode so no row is ever partially written. With `append=True` (vlnsafe.cli
uses this whenever a shard's output file already has rows) they append instead of
truncating, which is what makes an interrupted shard actually resumable -- cli.cmd_run
reads the existing episode_ids first and skips them. Both refuse any column not in
vlnsafe.records.schema. write_manifest dumps the reproducibility block (code, data, physics, seed).

Who reads it: vlnsafe.rollout (rows), vlnsafe.cli (manifest), scripts/merge_shards.py.

Must not import: habitat, torch.

Ported from the pre-release harness write_csv/write_json (126-138) and the manifest
assembly in the autopsy runner.
"""
import csv
import json
import os

from . import schema


class _StreamWriter(object):
    def __init__(self, path, columns, append=False):
        self._columns = columns
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        resuming = append and os.path.exists(path) and os.path.getsize(path) > 0
        if resuming:
            # appending rows of one schema under another's header corrupts the file
            # silently (a landed arm resumed after a schema change)
            with open(path, newline="") as f:
                header = next(csv.reader(f), None)
            if header != list(columns):
                raise ValueError(
                    "{} has a different header than the current schema; refusing to "
                    "append (write to a new --out)".format(path))
        self._fh = open(path, "a" if resuming else "w", newline="")
        self._csv = csv.writer(self._fh)
        if not resuming:
            self._csv.writerow(columns)
        self._fh.flush()

    def write(self, row):
        self._csv.writerow(schema.check_row(row, self._columns))
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class EpisodeWriter(_StreamWriter):
    def __init__(self, path, append=False):
        _StreamWriter.__init__(self, path, list(schema.EPISODE_COLUMNS), append=append)


class StepWriter(_StreamWriter):
    def __init__(self, path, append=False):
        _StreamWriter.__init__(self, path, list(schema.STEP_COLUMNS), append=append)


def write_manifest(path, fields):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(fields, f, indent=2, sort_keys=True)
        f.write("\n")
