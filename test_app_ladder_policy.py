"""Static safety checks for ladder exclusion at the Streamlit integration seam."""
from __future__ import annotations

import ast
from pathlib import Path

from wblot import project_io


APP_PATH = Path(__file__).with_name("app.py")


def _tree() -> ast.Module:
    return ast.parse(APP_PATH.read_text(encoding="utf-8"))


def _is_name(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Name) and node.id == value


def _is_session_key(node: ast.AST, key: str) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and _is_name(node.value, "S")
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == key
    )


def test_every_layout_to_measurement_iterator_receives_explicit_ladder_state():
    calls = [
        node for node in ast.walk(_tree())
        if isinstance(node, ast.Call)
        and _is_name(node.func, "rois_from_layout")
    ]

    assert len(calls) == 2
    assert all(
        len(call.args) == 4 and _is_session_key(call.args[3], "ladder_lanes")
        for call in calls
    )


def test_reference_lane_and_plot_order_use_quantifiable_lanes_only():
    tree = _tree()
    reference_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "selectbox"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Reference lane"
    ]
    assert len(reference_calls) == 1
    assert _is_name(reference_calls[0].args[1], "quant_lanes")

    plot_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "bar_chart"
    ]
    assert len(plot_calls) == 1
    order = next(keyword.value for keyword in plot_calls[0].keywords
                 if keyword.arg == "order")
    assert _is_name(order, "quant_lanes")


def test_project_schema_persists_explicit_ladder_lanes():
    assignment = next(
        node for node in _tree().body
        if isinstance(node, ast.Assign)
        and any(_is_name(target, "PROJECT_DEFAULTS") for target in node.targets)
    )
    assert isinstance(assignment.value, ast.Dict)
    keys = {
        key.value for key in assignment.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "ladder_lanes" in keys

    restored = project_io.load_project(project_io.project_json({
        "lanes": ["Ladder", "WT", "KO"],
        "ladder_lanes": ["Ladder"],
    }))
    assert restored["lanes"] == ["Ladder", "WT", "KO"]
    assert restored["ladder_lanes"] == ["Ladder"]


def test_all_export_tables_pass_through_the_lane_filter():
    source = APP_PATH.read_text(encoding="utf-8")
    assert 'lane_col="condition"' in source
    assert source.count("lane_roles.filter_lane_rows(") >= 8
