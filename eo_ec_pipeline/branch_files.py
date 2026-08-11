"""
Shared layout of the EO/EC branch tree, and the one safe way to list a branch's
.fif files.

data/branches/<condition>/{raw,denoised,upsampled}/ is persistent output: nothing
removes a file that is no longer wanted, and nothing adds one for a recording that
arrived after the last build. Globbing it blindly means a recording deleted from
data/input_* after a bad block was spotted keeps getting pooled into every result,
and a recording added since the last build is silently left out of it. Everything
downstream therefore lists branch files through branch_fifs(), which checks the
correspondence in both directions: every branch file must still have an input
recording, and every input recording must have branch output.
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


def subject_key(name: str) -> str:
    """The subject a recording belongs to, shared by that subject's EO and EC files.

    The design is paired (one EO and one EC block per subject), but nothing in the
    filesystem layout records the pairing: the condition lives in the folder, and
    the subject only in the filename. muse_to_fif.py routes on "open"/"closed"
    appearing in the export name, so the subject is what precedes that marker --
    "gary_closed_2026-08-09_raw" and "gary_open_2026-08-09_raw" both key to "gary".
    A block id built as "<condition>_<stem>" (run_eo_ec_test.py) is accepted too.
    Names without the marker fall back to their first underscore-separated token.
    """
    stem = Path(name).stem
    lowered = stem.lower()
    for prefix in ("ec_", "eo_"):
        if lowered.startswith(prefix):
            stem, lowered = stem[len(prefix):], lowered[len(prefix):]
    for marker in ("closed", "open"):
        cut = lowered.find(marker)
        if cut != -1:
            return lowered[:cut].strip("_- ") or lowered.split("_")[0]
    return lowered.split("_")[0]


def input_stems(condition: str) -> list[str]:
    """Filename stems of the recordings currently accepted for a condition."""
    return sorted(f.stem for f in Path(CONDITION_INPUT_DIRS[condition]).glob("*.fif"))


def branch_fifs(branch_dir, condition: str) -> list[Path]:
    """Branch .fif files, one per current input recording, or an error explaining why not.

    ZUNA may suffix its outputs, so a branch file is matched by input-stem prefix.
    Files with no matching input are stale (the recording was removed) and are
    skipped with a warning rather than silently pooled into the results. The reverse
    -- an input recording this branch was never built for, the usual result of adding
    a subject and re-running with --skip-build -- raises: dropping it would quietly
    compute the whole comparison over a subset of the recordings the operator
    believes are included. A missing or empty branch directory is the same failure at
    full scale (the branch vanishes from the ranking entirely) and raises too, so a
    caller never has to guess whether an empty list means "absent" or "uninteresting".
    """
    branch_dir = Path(branch_dir)
    found = sorted(branch_dir.glob("*.fif"))
    stems = input_stems(condition)

    if not stems:
        raise FileNotFoundError(
            f"No .fif files in {CONDITION_INPUT_DIRS[condition]} ({condition}), "
            f"but {branch_dir} holds {len(found)} file(s) — refusing to use stale output."
        )
    if not found:
        raise FileNotFoundError(
            f"No .fif files in {branch_dir}: this branch was never built for the "
            f"{len(stems)} {condition} recording(s) in {CONDITION_INPUT_DIRS[condition]} "
            "— rebuild the branches (re-run without --skip-build)."
        )

    current = [f for f in found if any(f.stem.startswith(s) for s in stems)]
    stale = [f.name for f in found if f not in current]
    if stale:
        print(f"WARNING: ignoring {len(stale)} stale file(s) in {branch_dir} "
              f"with no recording left in {CONDITION_INPUT_DIRS[condition]}: {stale}")

    unbuilt = [s for s in stems if not any(f.stem.startswith(s) for f in current)]
    if unbuilt:
        raise FileNotFoundError(
            f"{len(unbuilt)} recording(s) in {CONDITION_INPUT_DIRS[condition]} have no output in "
            f"{branch_dir}: {unbuilt} — rebuild the branches (re-run without --skip-build)."
        )
    return current
