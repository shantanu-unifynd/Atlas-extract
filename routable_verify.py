"""P0 regression harness — measure routable-% for every reference plan.

Usage:
    PYTHONPATH=. python routable_verify.py            # all reference plans
    PYTHONPATH=. python routable_verify.py <plan>...  # specific files

Prints one row per plan: mode, shops, routable ok/far/cross, pct, skeleton
pieces, main-piece share. Run before AND after any pipeline change; a drop on
a previously-good plan is a regression.
"""
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

from app.pipeline.routable import measure

HERE = pathlib.Path(__file__).parent
ASSETS = HERE.parent / "assets"
REFERENCE = [
    HERE / "tests" / "fixtures" / "wtc-mall.svg",
    HERE / "tests" / "fixtures" / "FF.pdf",
    ASSETS / "0001.svg",
    ASSETS / "0002.svg",
    ASSETS / "PHOENIX PALASSIO TRADING FLOOR PLAN - 23-01-26.pdf",
]


def main(argv):
    paths = [pathlib.Path(p) for p in argv] if argv else REFERENCE
    print(f"{'plan':<44} {'mode':<7} {'shops':>5} {'ok':>4} {'far':>4} {'cross':>5} "
          f"{'pct':>6} {'pieces':>6} {'main%':>6}")
    for p in paths:
        if not p.exists():
            print(f"{p.name:<44} MISSING")
            continue
        try:
            r = measure(str(p))
        except Exception as e:
            print(f"{p.name:<44} ERROR {e}")
            continue
        print(f"{p.name[:43]:<44} {r['mode']:<7} {r['shops']:>5} {r['ok']:>4} "
              f"{r['far']:>4} {r['cross']:>5} {r['pct']:>5.1f}% {r['pieces']:>6} "
              f"{r['main_frac'] * 100:>5.0f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
