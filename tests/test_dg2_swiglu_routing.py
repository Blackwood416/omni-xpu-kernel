"""Only the measured A770 H3 layout may enter the existing exact kernel."""
from types import SimpleNamespace

import pytest

import omni_xpu_kernel
from omni_xpu_kernel import int8


@pytest.mark.parametrize("target,core,shape,weight,expected", [
    ("dg2", "dg2", (16473, 28672), (5376, 14336), True),
    ("dg2", "bmg", (16473, 28672), (5376, 14336), False),
    ("dg2", "dg2", (16472, 28672), (5376, 14336), False),
    ("dg2", "dg2", (16473, 28672), (4096, 14336), False),
    ("ptl-h", "ptl-h", (16473, 28672), (5376, 14336), False),
    ("bmg", "bmg", (3, 512), (96, 256), True),
])
def test_exact_swiglu_target_and_layout(monkeypatch, target, core, shape, weight, expected):
    monkeypatch.setattr(omni_xpu_kernel, "__xpu_target__", target)
    monkeypatch.setattr(omni_xpu_kernel, "core_aot_target", lambda: core)
    assert int8._is_supported_h3_swiglu_input(
        SimpleNamespace(shape=shape), SimpleNamespace(shape=weight)) is expected


def test_dg2_does_not_enable_bmg_memory_policies(monkeypatch):
    monkeypatch.setattr(omni_xpu_kernel, "__xpu_target__", "dg2")
    monkeypatch.setattr(omni_xpu_kernel, "core_aot_target", lambda: "dg2")
    assert not int8._is_supported_h3_swiglu_target()
