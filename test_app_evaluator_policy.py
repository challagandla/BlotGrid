"""Static policy checks for the optional Streamlit evaluator-review surface."""
from __future__ import annotations

import ast
from pathlib import Path


APP_PATH = Path(__file__).with_name("app.py")


def _labelled_calls(tree: ast.AST, label: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == label
    ]


def _disabled_by_evaluator_lock(call: ast.Call) -> bool:
    return any(
        keyword.arg == "disabled"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "evaluation_task_locked"
        for keyword in call.keywords
    )


def test_evaluator_task_locks_detector_and_metadata_mutations():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    locked_labels = {
        "Targets (one per line, top→bottom)",
        "Loading control (optional)",
        "Lanes / conditions (one per line, left→right)",
        "Ladder / marker lanes (excluded from quantification)",
        "Detection backend",
        "AI checkpoint path",
        "Minimum AI score",
        "Review below",
        "AI device",
        "GelGenie ONNX model path",
        "GelGenie normalization",
        "Band probability",
        "GelGenie device",
        "SAM 2.1 checkpoint path",
        "SAM device",
        "SAM context around ROI",
        "Band polarity",
        "Classical / recovery sensitivity",
        "Apply these labels",
        "Detect all bands",
        "Use even grid",
        "Refine this ROI position with SAM 2.1",
        "Width (px)",
        "Height (px)",
        "Apply size to every band",
    }
    for label in locked_labels:
        calls = _labelled_calls(tree, label)
        assert calls, f"missing expected evaluator-sensitive control: {label}"
        assert all(
            _disabled_by_evaluator_lock(call) for call in calls
        ), f"evaluator-sensitive control is not locked: {label}"

    editor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "edit_rois"
    ]
    assert len(editor_calls) == 1
    allow_resize = next(
        keyword.value
        for keyword in editor_calls[0].keywords
        if keyword.arg == "allow_resize"
    )
    assert isinstance(allow_resize, ast.UnaryOp)
    assert isinstance(allow_resize.op, ast.Not)
    assert isinstance(allow_resize.operand, ast.Name)
    assert allow_resize.operand.id == "evaluation_task_locked"


def test_evaluator_task_omits_alignment_ui_and_guards_ocr_autofill():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    alignment_calls = _labelled_calls(
        tree, "Straighten scan with alignment grid"
    )
    assert len(alignment_calls) == 1
    alignment_call = alignment_calls[0]
    alignment_is_guarded = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "evaluation_task_locked"
        and any(
            alignment_call in ast.walk(branch)
            for branch in node.orelse
        )
        for node in ast.walk(tree)
    )
    assert alignment_is_guarded

    autofill_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_queue_detected_metadata"
    ]
    assert autofill_calls
    guarded_autofill = False
    for conditional in (
        node for node in ast.walk(tree) if isinstance(node, ast.If)
    ):
        if not any(
            isinstance(name, ast.Name)
            and name.id == "evaluation_task_locked"
            for name in ast.walk(conditional.test)
        ):
            continue
        if any(
            call in ast.walk(statement)
            for statement in conditional.body
            for call in autofill_calls
        ):
            guarded_autofill = True
            break
    assert guarded_autofill


def test_clear_all_blots_discards_public_source_lineage():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    clear_calls = _labelled_calls(tree, "Clear all blots")
    assert len(clear_calls) == 1
    clear_call = clear_calls[0]
    clear_branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and clear_call in ast.walk(node.test)
    )
    cleared_string_literals = {
        node.value
        for statement in clear_branch.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "source_lineage" in cleared_string_literals
