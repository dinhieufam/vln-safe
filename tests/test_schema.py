"""The released log contract is self-consistent and stable. No habitat."""
import pytest

from vlnsafe.records import schema


def test_column_lists_have_no_duplicates():
    for cols in (schema.EPISODE_COLUMNS, schema.STEP_COLUMNS):
        assert len(cols) == len(set(cols))


def test_decision_only_columns_are_a_subset_of_step_columns():
    assert set(schema.DECISION_ONLY_STEP_COLUMNS).issubset(set(schema.STEP_COLUMNS))


def test_check_row_orders_and_rejects_off_schema_rows():
    cols = ("a", "b", "c")
    assert schema.check_row({"c": 3, "a": 1, "b": 2}, cols) == [1, 2, 3]
    with pytest.raises(KeyError):
        schema.check_row({"a": 1, "b": 2}, cols)          # missing c
    with pytest.raises(KeyError):
        schema.check_row({"a": 1, "b": 2, "c": 3, "d": 4}, cols)  # extra d


def test_unavailable_is_a_nonempty_string():
    assert isinstance(schema.UNAVAILABLE, str) and schema.UNAVAILABLE
