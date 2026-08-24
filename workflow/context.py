# context.py
#
# The binding for one run.
#
# `.wf/` is the durable state across invocations; a Context is only what a
# single invocation resolves paths against. It is built once at command entry
# and is fully immutable -- no field is written while a recipe runs.
#
# Steps derive every path they touch from a Context and never receive a value
# from the step before them. That is what lets a step be skipped as
# already-done without stranding the step after it.

from dataclasses import dataclass
from pathlib import Path

from workflow import config, names


@dataclass(frozen=True)
class Context:
    # Bundle coordinates: what a lifecycle step resolves its paths against.
    root: Path            # the workflow root -- the directory *containing* .wf
    phase: str            # p1 | p2 | p3
    force: bool = False   # ignore is_done and overwrite
    bundle_name: str = ""  # eval directory name; empty for non-bundle reads

    # Query parameters: what a corpus read needs and a bundle operation does not.
    selector: str = "all"        # handed to select() by steps that read a slot
    dest: Path | None = None     # where a non-bundle read places its result
    results_dir: Path | None = None  # optional corpus outside the workflow slot
    pairs_path: Path | None = None   # optional identity mask for a corpus read
    pmin: float = 0.9
    prange: float = 0.1

    @property
    def bundle_dir(self) -> Path:
        if not self.bundle_name:
            raise ValueError("context has no bundle: bundle name is empty")
        return config.path(self.root, [self.phase, "eval"]) / self.bundle_name

    def artifact(self, classifier: str, kind: str) -> Path:
        """A rendered artifact path inside this bundle."""
        return self.bundle_dir / names.artifact(
            self.bundle_name, classifier, kind)
