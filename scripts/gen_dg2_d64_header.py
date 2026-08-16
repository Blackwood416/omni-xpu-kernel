"""Generate the DG2 D64 DPAS attention header from the D128 v4.1 source.

The D64 port keeps the exact v4.1 structure (packed-Q/K/V + DPAS + SLM
staging, WG=32) and only changes the head dimension:

  DCHUNKS 8 -> 4
  row width / accumulator 128 -> 64
  per-tile packed words BN*64 -> BN*32 (SLM bytes BN*64*4 -> BN*32*4)
  K/V head stride 128 -> 64
  staging loop BN/32 -> BN/64
  SCALE 1/sqrt(128) -> 1/sqrt(64) = 0.125
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "omni_xpu_kernel" / "lgrf_uni" / "single_kernels"
SOURCE = SRC / "flash.attn.b.mha.dg2.dpas4.h"
TARGET = SRC / "flash.attn.b.mha.dg2.dpas4.d64.h"

REPLACEMENTS = [
    ("namespace dg2v4 {", "namespace dg2v4d64 {"),
    ("}  // namespace dg2v4", "}  // namespace dg2v4d64"),
    ("constexpr int DCHUNKS = 8;", "constexpr int DCHUNKS = 4;"),
    ("RPT * 128", "RPT * 64"),
    ("r * 128 + c * 16", "r * 64 + c * 16"),
    ("select<128, 1>(r * 128)", "select<64, 1>(r * 64)"),
    ("BN * 64 * 4", "BN * 32 * 4"),
    ("BN * 64", "BN * 32"),
    ("(BN) * 64", "(BN) * 32"),
    ("headKv * 128", "headKv * 64"),
    ("headIdx) * 128", "headIdx) * 64"),
    ("BN / 32", "BN / 64"),
    ("0.08838834764831844f", "0.125f"),
]


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        if old not in text:
            raise SystemExit(f"pattern not found: {old!r}")
        text = text.replace(old, new)
    text = (
        "// DG2-native Flash Attention v4.1 D64 port (MiniMax H3 VideoVAE).\n"
        "// Generated from flash.attn.b.mha.dg2.dpas4.h with DCHUNKS=4 and\n"
        "// row width 64; see scripts/gen_dg2_d64_header.py for the mapping.\n"
        "// Same A770 rules: WG=32, zero spill, fp32 DPAS accumulation.\n\n"
        + text
    )
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
