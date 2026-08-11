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
    appearing in the export name, so the subject is the rest of the name once that
    marker is removed --
    "gary_closed_2026-08-09_raw" and "gary_open_2026-08-09_raw" both key to "gary",
    and "closed_gary_2026-08-09" and "open_gary_2026-08-09" both key to
    "gary_2026-08-09". The README only requires the marker to appear somewhere in the
    name, so a leading marker has to be handled: taking the text before it and falling
    back to the first underscore token would return the marker itself, keying every EC
    recording to "closed" and every EO one to "open" -- two condition-shaped
    pseudo-subjects that each hold a single condition, so every EC-EO contrast is NaN.
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
        if cut == -1:
            continue
        before = lowered[:cut].strip("_- ")
        if before:
            return before
        after = lowered[cut + len(marker):].strip("_- ")
        if after:
            return after
        raise ValueError(
            f"{name!r} is nothing but the condition marker {marker!r}: it names no "
            "subject, so its EO and EC recordings could never be paired."
        )
    return lowered.split("_")[0]


def input_stems(condition: str) -> list[str]:
    """Filename stems of the recordings currently accepted for a condition."""
    return sorted(f.stem for f in Path(CONDITION_INPUT_DIRS[condition]).glob("*.fif"))


def branch_fifs(branch_dir, condition: str) -> list[Path]:
    """Branch .fif files, one per current input recording, or an error explaining why not.

    ZUNA may suffix its outputs, so a branch file is matched by input-stem prefix, to
    the *longest* input stem it starts with: with inputs "gary_open" and "gary_open_2",
    "gary_open_2"'s output prefix-matches both, and crediting it to "gary_open" would
    let the shorter recording pass as built while its own output is missing.
    Files with no matching input are stale (the recording was removed) and are
    skipped with a warning rather than silently pooled into the results. The reverse
    -- an input recording this branch was never built for, the usual result of adding
    a subject and re-running with --skip-build -- raises: dropping it would quietly
    compute the whole comparison over a subset of the recordings the operator
    believes are included. A missing or empty branch directory is the same failure at
    full scale (the branch vanishes from the ranking entirely) and raises too, so a
    caller never has to guess whether an empty list means "absent" or "uninteresting".

    Two branch files for one recording raise as well. Callers treat the result as one
    file per recording (run_eo_ec_test.py gives each its own block id, and
    classify.classify makes each block id a cross-validation group), so a second
    output -- an MNE split part "<stem>-1.fif", a leftover from an earlier run under a
    different suffix -- would enter the pool as a second, independent recording,
    inflating both the group count and that subject's weight.
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

    matched: dict[str, list[Path]] = {s: [] for s in stems}
    stale = []
    for f in found:
        # longest match wins: a stem that is itself a prefix of another stem must not
        # claim the longer recording's output.
        owner = max((s for s in stems if f.stem.startswith(s)), key=len, default=None)
        if owner is None:
            stale.append(f.name)
        else:
            matched[owner].append(f)
    if stale:
        print(f"WARNING: ignoring {len(stale)} stale file(s) in {branch_dir} "
              f"with no recording left in {CONDITION_INPUT_DIRS[condition]}: {stale}")

    unbuilt = [s for s in stems if not matched[s]]
    if unbuilt:
        raise FileNotFoundError(
            f"{len(unbuilt)} recording(s) in {CONDITION_INPUT_DIRS[condition]} have no output in "
            f"{branch_dir}: {unbuilt} — rebuild the branches (re-run without --skip-build)."
        )
    duplicated = {s: [f.name for f in fs] for s, fs in matched.items() if len(fs) > 1}
    if duplicated:
        raise FileNotFoundError(
            f"{len(duplicated)} recording(s) in {CONDITION_INPUT_DIRS[condition]} have more than "
            f"one output in {branch_dir}: {duplicated} — each would be pooled as a separate "
            "recording. Delete the extra file(s), or clear the branch directory and rebuild "
            "(re-run without --skip-build)."
        )
    return [matched[s][0] for s in stems]
