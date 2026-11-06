"""python -m egowhale <episode.hdf5> [out_dir] [--steps NAME ...] [--inpaint-model sttn|propainter]"""

from __future__ import annotations

import argparse
from pathlib import Path

from egowhale.action.approach import Approach
from egowhale.action.base_ik import BaseIK
from egowhale.action.retarget import Retarget
from egowhale.annotate import Annotate
from egowhale.paths import default_out
from egowhale.quality.curate import Curate
from egowhale.quality.vlm_audit import VlmAudit
from egowhale.step import run
from egowhale.visual.composite import Composite
from egowhale.visual.depth import Depth
from egowhale.visual.inpaint import DEFAULT_MODEL, MODELS, Inpaint
from egowhale.visual.render import Render
from egowhale.visual.segment import Segment

STEPS = (Annotate, Retarget, Segment, Inpaint, Depth, BaseIK, Approach, Render, Composite, Curate, VlmAudit)


def main() -> None:
    names = [step.name for step in STEPS]
    parser = argparse.ArgumentParser(prog="python -m egowhale", description="Run the pipeline on one episode in this process.")
    parser.add_argument("episode", type=Path, help="EgoDex HDF5 file with a sibling .mp4")
    parser.add_argument("out", type=Path, nargs="?", help="output directory; default outputs/<task>/<episode>")
    parser.add_argument("--steps", nargs="+", choices=names, metavar="NAME", help=f"run only these, in pipeline order: {' '.join(names)}")
    parser.add_argument("--inpaint-model", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--annotate-mode", choices=("off", "description", "subtasks", "both"), default="both")
    parser.add_argument("--vlm-model", help="override vlm.model (or set EGOWHALE_VLM_MODEL)")
    args = parser.parse_args()
    src = args.episode.expanduser().resolve()
    dst = args.out.expanduser().resolve() if args.out else default_out(src)
    config = {"annotate": {"mode": args.annotate_mode}}
    if args.vlm_model:
        config["vlm"] = {"model": args.vlm_model}
    chosen = [step for step in STEPS if args.steps is None or step.name in args.steps]
    run(
        src,
        dst,
        tuple(
            Inpaint(args.inpaint_model)
            if step is Inpaint
            else step(config)
            if step in (Annotate, VlmAudit)
            else step
            for step in chosen
        ),
    )
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
