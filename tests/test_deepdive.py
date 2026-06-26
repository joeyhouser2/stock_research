"""Tests for deepdive grid assembly (the network-free parts)."""

from stock_research import deepdive


def test_grid_columns_default_is_unchanged():
    assert deepdive.grid_columns(False) == deepdive.GRID_COLUMNS
    assert deepdive.grid_columns(False) is not deepdive.GRID_COLUMNS  # a copy


def test_grid_columns_inserts_sim_after_prob_otm():
    cols = deepdive.grid_columns(True)
    # The three probabilities sit together, with touch right after the terminal sim.
    assert cols.index("sim_otm") == cols.index("prob_otm") + 1
    assert cols.index("sim_touch") == cols.index("sim_otm") + 1
    assert cols.index("delta") == cols.index("sim_touch") + 1
    # Every base column survives, plus exactly the two sim columns.
    assert set(cols) == set(deepdive.GRID_COLUMNS) | set(deepdive.SIM_COLUMNS)
    assert len(cols) == len(deepdive.GRID_COLUMNS) + 2
