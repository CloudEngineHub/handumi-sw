from __future__ import annotations

import pytest

from handumi.scripts.inpaint_context import parse_episodes


def test_a_single_episode():
    assert parse_episodes("3") == [3]


def test_a_comma_separated_list():
    assert parse_episodes("0,2,5") == [0, 2, 5]


def test_a_range():
    assert parse_episodes("0-4") == [0, 1, 2, 3, 4]


def test_ranges_and_singles_together():
    assert parse_episodes("0-3,7,10-11") == [0, 1, 2, 3, 7, 10, 11]


def test_duplicates_collapse_and_order_is_stable():
    """A repeated episode must not be generated -- and paid for -- twice."""
    assert parse_episodes("5,1,5,1") == [1, 5]


def test_empty_selection_is_refused():
    with pytest.raises(Exception):
        parse_episodes(",  ,")
