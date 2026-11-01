"""CLI: python -m app.cli <plan.svg|plan.pdf> <outdir>"""
import json
import pathlib
import sys

from app.pipeline.run import extract


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src, outdir = sys.argv[1], pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    result = extract(src)
    (outdir / "report.json").write_text(json.dumps(result["report"], indent=1))
    (outdir / "atlas.json").write_text(json.dumps(result["atlas"], indent=1))
    print(json.dumps(result["report"], indent=1))


if __name__ == "__main__":
    main()
