# steps/p2_extract.py
#
# Parse the retrieved enex into YES and NO pair sets, produced into the bundle
# directory. Two steps, one implementation: the kind is the only difference, so
# they are instances rather than two near-identical modules. The runner only
# needs NAME/outputs/run_step, not a module.

import subprocess

from pathlib import Path

from workflow import fs, log, setops
from workflow.steps import p2_retrieve


def _parse_note_files(paths: list[Path], kind: str) -> list[Path]:
    parsed = []
    for path in paths:
        out_path = Path(f"/tmp/{path.name}.parsed")
        with out_path.open("w") as f:
            subprocess.run(["note", "--parse-file", str(path), "--type", kind,
                            "--lines"], stdout=f, check=True)
        parsed.append(out_path)
    return parsed


class _Extract:
    def __init__(self, kind: str):
        self.kind = kind
        self.NAME = f"extract_{kind}"

    def inputs(self, ctx) -> list[Path]:
        return sorted(p2_retrieve.enex_dir(ctx).glob("*.enex"))

    def outputs(self, ctx) -> list[Path]:
        return [ctx.artifact("p2", self.kind)]

    def run_step(self, ctx) -> None:
        enex = self.inputs(ctx)
        fs.raise_if_not_dir(p2_retrieve.enex_dir(ctx))
        parsed = _parse_note_files(enex, self.kind)
        setops.merge(parsed, self.outputs(ctx)[0])
        log.info(f"extracted {self.kind.upper()} pairs from {len(enex)} note part(s)")


YES = _Extract("yes")
NO = _Extract("no")
