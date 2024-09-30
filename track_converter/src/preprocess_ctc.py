from pathlib import Path

import pandas as pd


def validate_tracks_shape_dtypes(tracks: pd.DataFrame) -> None:
    """
    Validate the shape (number of rows / columns) and data types / ranges of columns.

    This will also add a final right-censoring flag column (defaulting to all zero) if it doesn't exist.
    """
    nrows = tracks.shape[0]
    ncols = tracks.shape[1]

    if ncols not in (4, 5):
        msg = "input file must have 4 or 5 columns"
        raise ValueError(msg)

    if nrows == 0:
        msg = "input file must contain at least one track"
        raise ValueError(msg)

    # add right-censoring column (if it doesn't exist), defaulting to all zero
    if ncols == 4:
        tracks["R"] = 0

    # Re-name columns to LBEPR for easy access
    tracks.columns = ["L", "B", "E", "P", "R"]

    # Check data type and ranges of columns
    cols_are_integers = tracks.apply(pd.api.types.is_integer_dtype, axis=0)
    if not cols_are_integers.all():
        msg = "all columns must contain integer values"
        raise ValueError(msg)

    if not (tracks["L"] > 0).all():
        msg = "all track labels (first column) must be greater than zero"
        raise ValueError(msg)

    if not tracks["L"].is_unique:
        msg = "all track labels (first column) must be unique"
        raise ValueError(msg)

    if not (tracks["P"] >= 0).all():
        msg = "all track parent labels (fourth column) must be greater than or equal to zero"
        raise ValueError(msg)

    if not tracks["R"].isin((0, 1)).all():
        msg = "all right-censoring flags (last column) must be 0 or 1"
        raise ValueError(msg)


def validate_track_begin_end_frames(tracks: pd.DataFrame) -> None:
    """
    Validate the begin and end frames of the tracks.

    Begin frame (B) must be <= end frame (E), unless it is by a single frame.
    E can be one less than B if the cell existed but was never observed.
    """
    tracks_begin_after_end = tracks.loc[tracks["B"] > tracks["E"], :]
    if tracks_begin_after_end.empty:
        return

    if not (tracks_begin_after_end["B"] - tracks_begin_after_end["E"] == 1).all():
        msg = "The tracks start frame must be <= the tracks end frame, unless it is by a single frame."
        raise ValueError(msg)


def validate_n_daughters(tracks: pd.DataFrame) -> None:
    """
    Validate the number of daughters produced by each cell, as well as some other simple checks for the parent id.

    Each mother-cell should either have two daughters or none (right-censored or dying).
    """
    # All parent ids (P) should match a track label (L) or be 0
    parents_in_labels = tracks["P"].isin(tracks["L"])
    parents_are_zero = tracks["P"] == 0
    if not (parents_in_labels | parents_are_zero).all():
        msg = "Track parents (fourth column) must match a track label (first column), or be set to zero"
        raise ValueError(msg)

    # The parent id (P) of a track can't be equal to its label (L)
    if (tracks["P"] == tracks["L"]).any():
        msg = "A tracks parent (fourth column) can't be equal to its label (first column)"
        raise ValueError(msg)

    # Each parent id should appear twice (i.e. two daughters), except for zero which can occur any number of times
    parent_counts = tracks["P"].value_counts()
    two_daughters = parent_counts == 2
    zero_label = parent_counts.index.to_series() == 0
    if not (two_daughters | zero_label).all():
        msg = "Tracks must have 2 daughters, or None"
        raise ValueError(msg)


def validate_mother_daughter_frames(tracks: pd.DataFrame) -> None:
    """Validate that both daughter cells appear one frame after the mother's last frame."""
    # ignore parent ids of 0 (this means the parent is unknown)
    tracks_with_parents = tracks.loc[tracks["P"] != 0, :]

    # For each track, its begin frame (B) must be one more than its parent's end frame (E)
    tracks_with_parents = tracks_with_parents.merge(
        tracks[["L", "E"]], how="left", left_on="P", right_on="L", suffixes=("_track", "_parent")
    )
    tracks_with_parents = tracks_with_parents[["L_track", "P", "B", "E_parent"]]

    if not (tracks_with_parents["B"] - tracks_with_parents["E_parent"] == 1).all():
        msg = "Daughter cells must appear one frame after the mother's last frame"
        raise ValueError(msg)


def preprocess_ctc_file(input_ctc_filepath: Path, output_ctc_filepath: Path) -> None:
    """
    Preprocess Cell Tracking Challenge (CTC) format files.

    Expects files in the standard CTC format, optionally with an additional column indicating right censoring.
    Columns are L B E P (R):
    L - a unique label of the track (any positive number, not zero or negative)
    B - the frame where the track begins (integer)
    E - the frame where the track ends (integer)
    P - label of the parent track (0 is used when the parent is unknown)
    R - right censoring flag (1=right-censored, 0=not). Note a 0 only means it is not manually declared as
    right-censored. It will still be considered right-censored by the processing code, if the last observed frame
    coincides with the end of the movie.
    """
    tracks = pd.read_table(input_ctc_filepath, sep=r"\s+", header=None)
    validate_tracks_shape_dtypes(tracks)
    validate_track_begin_end_frames(tracks)
    validate_n_daughters(tracks)
    validate_mother_daughter_frames(tracks)

    # save new file
    tracks.to_csv(output_ctc_filepath, sep=" ", header=False, index=False)
