"""
Shared layout of the EO/EC branch tree, and the one safe way to list a branch's
.fif files.

data/branches/<condition>/{raw,denoised,upsampled}/ is persistent output: nothing
removes a file that is no longer wanted. Globbing it blindly means a recording
that was deleted from data/input_* after a bad block was spotted keeps getting
pooled into every result. Everything downstream therefore lists branch files
through branch_fifs(), which keeps only the files that still have a matching
recording in the condition's input directory and says out loud what it dropped.
"""

from pathlib import Path

CONDITION_INPUT_DIRS = {"EC": "data/input_ec", "EO": "data/input_eo"}

# branch name -> subdirectory under data/branches/<condition>/
# hybrid is skipped everywhere: identical to full_reconstruction for denoised, and
# full_reconstruction is used for upsampled too, for a consistent comparison.
BRANCH_SUBDIR = {
    "raw": "raw",
    "denoised": "denoised/full_reconstruction",
    "upsampled": "upsampled/full_reconstruction",
}


def input_stems(condition: str) -> list[str]:
    """Filename stems of the recordings currently accepted for a condition."""
    return sorted(f.stem for f in Path(CONDITION_INPUT_DIRS[condition]).glob("*.fif"))


def branch_fifs(branch_dir, condition: str) -> list[Path]:
    """Branch .fif files that still correspond to a current input recording.

    ZUNA may suffix its outputs, so a branch file is matched by input-stem prefix.
    Files with no matching input are stale (the recording was removed) and are
    skipped with a warning rather than silently pooled into the results.
    """
    branch_dir = Path(branch_dir)
    found = sorted(branch_dir.glob("*.fif"))
    if not found:
        return []

    stems = input_stems(condition)
    if not stems:
        raise FileNotFoundError(
            f"No .fif files in {CONDITION_INPUT_DIRS[condition]} ({condition}), "
            f"but {branch_dir} still holds {len(found)} file(s) — refusing to use stale output."
        )

    current = [f for f in found if any(f.stem.startswith(s) for s in stems)]
    stale = [f.name for f in found if f not in current]
    if stale:
        print(f"WARNING: ignoring {len(stale)} stale file(s) in {branch_dir} "
              f"with no recording left in {CONDITION_INPUT_DIRS[condition]}: {stale}")
    if not current:
        raise FileNotFoundError(
            f"None of the {len(found)} file(s) in {branch_dir} match a recording in "
            f"{CONDITION_INPUT_DIRS[condition]} ({stems}) — rebuild the branches."
        )
    return current
