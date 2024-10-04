import logging
from pathlib import Path

import pandas as pd

from track_converter.src.utils import discard_related_cells

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_tracks_shape_dtypes(tracks: pd.DataFrame) -> None:
    """
    Validate the shape (number of rows / columns) and data types / ranges of columns.

    This will also add a final right-censoring flag column (defaulting to all zero) if it doesn't exist.
    """
    nrows = tracks.shape[0]
    ncols = tracks.shape[1]

    if ncols not in (4, 5):
        msg = "input file must have 4 or 5 columns"
        logger.error(msg)
        raise ValueError(msg)

    if nrows == 0:
        msg = "input file must contain at least one tracked cell"
        logger.error(msg)
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
        logger.error(msg)
        raise ValueError(msg)

    if not (tracks["L"] > 0).all():
        msg = "all tracked cell labels (first column) must be greater than zero"
        logger.error(msg)
        raise ValueError(msg)

    if not tracks["L"].is_unique:
        msg = "all tracked cell labels (first column) must be unique"
        logger.error(msg)
        raise ValueError(msg)

    if not (tracks["P"] >= 0).all():
        msg = "all parent labels (fourth column) must be greater than or equal to zero"
        logger.error(msg)
        raise ValueError(msg)

    if not tracks["R"].isin((0, 1)).all():
        msg = "all right-censoring flags (last column) must be 0 or 1"
        logger.error(msg)
        raise ValueError(msg)


def validate_cell_begin_end_frames(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the begin and end frames of the tracked cells.

    Begin frame (B) must be <= end frame (E), unless it is by a single frame.
    E can be one less than B if the cell existed but was never observed.
    """
    passed_msg = "Checking cell begin/end frames... Passed"

    cells_begin_after_end = tracks.loc[tracks["B"] > tracks["E"], :]
    if cells_begin_after_end.empty:
        logger.info(passed_msg)
        return tracks

    cells_are_invalid = cells_begin_after_end["B"] - cells_begin_after_end["E"] != 1
    if cells_are_invalid.any():
        invalid_labels = cells_begin_after_end.loc[cells_are_invalid, "L"]
        msg = (
            f"All cell start frames must be <= their end frame, unless it is by a single frame. Removing all cells "
            f"connected to invalid labels: {invalid_labels.to_numpy()}"
        )
        logger.warning(msg)
        return discard_related_cells(tracks, invalid_labels)

    logger.info(passed_msg)
    return tracks


def validate_parents_in_labels(tracks: pd.DataFrame) -> pd.DataFrame:
    """Validate that all parent labels (P) appear in the cell labels (L) or are 0."""
    parents_in_labels = tracks["P"].isin(tracks["L"])
    parents_are_zero = tracks["P"] == 0
    cells_are_valid = parents_in_labels | parents_are_zero

    if not cells_are_valid.all():
        invalid_labels = tracks.loc[~cells_are_valid, "L"]
        msg = (
            f"Parent labels (fourth column) must match a cell label (first column), or be set to zero. "
            f"Removing all cells connected to invalid labels: {invalid_labels.to_numpy()}"
        )
        logger.warning(msg)
        return discard_related_cells(tracks, invalid_labels)

    logger.info("Checking parent labels (P) appear in cell labels (L)... Passed")
    return tracks


def validate_parent_label_unequal(tracks: pd.DataFrame) -> pd.DataFrame:
    """Validate that no cells have a parent label (P) equal to their own label (L)."""
    cells_are_invalid = tracks["P"] == tracks["L"]
    if cells_are_invalid.any():
        invalid_labels = tracks.loc[cells_are_invalid, "L"]
        msg = (
            f"A cell's parent (fourth column) can't be equal to its label (first column)."
            f"Removing all cells connected to invalid labels: {invalid_labels.to_numpy()}"
        )
        logger.warning(msg)
        return discard_related_cells(tracks, invalid_labels)

    logger.info("Checking no cells have a parent label (P) equal to their own label (L)... Passed")
    return tracks


def correct_missing_daughters(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Create a second daughter for any mother cells that only have one.

    This daughter will have (L B E P R) of:
    (next available label - frame after mother ends - frame mother ends - mother label - 1)
    """
    parent_counts = tracks["P"].value_counts()
    parents_with_one_daughter = parent_counts.index.to_series()[parent_counts == 1]

    if len(parents_with_one_daughter) > 0:
        msg = f"Cells with label {parents_with_one_daughter.to_numpy()} only have one daughter - creating a second"
        logger.warning(msg)

        max_label = tracks["L"].max()

        for parent in parents_with_one_daughter:
            max_label += 1
            parent_end = tracks.loc[tracks["L"] == parent, "E"].to_numpy()[0]
            tracks.loc[len(tracks.index)] = [max_label, parent_end + 1, parent_end, parent, 1]

    return tracks


def validate_n_daughters(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the number of daughters produced by each cell.

    Each mother-cell should either have two daughters or none (right-censored or dying).
    """
    # Each parent label should appear twice (i.e. two daughters), except for zero which can occur any number of times
    parent_counts = tracks["P"].value_counts()
    cells_are_invalid = (parent_counts != 2) & (parent_counts.index.to_series() != 0)

    if cells_are_invalid.any():
        invalid_labels = parent_counts.index.to_series()[cells_are_invalid]
        msg = (
            f"Cells must have 2 daughters, or None. "
            f"Removing all cells connected to invalid labels: {invalid_labels.to_numpy()}"
        )
        logger.warning(msg)
        return discard_related_cells(tracks, invalid_labels)

    logger.info("Checking cells have two daughters or no daughters... Passed")
    return tracks


def correct_late_daughters(tracks: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all pairs of daughter cells have the same begin frame (B).

    Back-date any late daughters to the begin time of the earlier daughter.
    """
    # ignore parent ids of 0 (this means the parent is unknown)
    has_parents = tracks["P"] != 0
    tracks_with_parents = tracks.loc[has_parents, :]

    different_begin_frame = tracks_with_parents.groupby("P")["B"].nunique() != 1
    if different_begin_frame.any():
        msg = (
            f"Daughters with parents {different_begin_frame.index[different_begin_frame].to_numpy()} "
            f"don't have the same begin frame (B) - setting to minimum"
        )
        logger.warning(msg)

        # Set all cells with parents to the minimum begin frame
        begin_min = tracks_with_parents.groupby("P")["B"].transform("min")
        tracks.loc[has_parents, "B"] = begin_min

    return tracks


def validate_mother_daughter_frames(tracks: pd.DataFrame) -> pd.DataFrame:
    """Validate that both daughter cells appear one frame after the mother's last frame."""
    # ignore parent ids of 0 (this means the parent is unknown)
    cells_with_parents = tracks.loc[tracks["P"] != 0, :]

    cells_with_parents = cells_with_parents.merge(
        tracks[["L", "E"]], how="left", left_on="P", right_on="L", suffixes=("_track", "_parent")
    )
    cells_with_parents = cells_with_parents[["L_track", "P", "B", "E_parent"]]

    # For each tracked cell, its begin frame (B) must be one more than its parent's end frame (E)
    cells_are_invalid = cells_with_parents["B"] - cells_with_parents["E_parent"] != 1
    if cells_are_invalid.any():
        invalid_labels = cells_with_parents.loc[cells_are_invalid, "L_track"]
        msg = (
            f"Daughter cells must appear one frame after the mother's last frame. "
            f"Removing all cells connected to invalid labels: {invalid_labels.to_numpy()}"
        )
        logger.warning(msg)
        return discard_related_cells(tracks, invalid_labels)

    logger.info("Checking daughter cells appear one frame after the mother's last frame... Passed")
    return tracks


def validate_right_censored_daughters(tracks: pd.DataFrame) -> pd.DataFrame:
    """Validate that all right-censored cells have no daughters."""
    right_censored_labels = tracks.loc[tracks["R"] == 1, "L"]
    cells_are_invalid = tracks["P"].isin(right_censored_labels)

    if cells_are_invalid.any():
        invalid_labels = tracks.loc[cells_are_invalid, "L"]
        msg = (
            f"Right-censored cells should have no daughters."
            f"Removing all cells connected to invalid labels: {invalid_labels.to_numpy()}"
        )
        logger.warning(msg)
        return discard_related_cells(tracks, invalid_labels)

    logger.info("Checking right-censored cells have no daughters... Passed")
    return tracks


def preprocess_ctc_file(
    input_ctc_filepath: Path,
    output_ctc_filepath: Path,
    fix_late_daughters: bool = False,
    fix_missing_daughters: bool = False,
) -> None:
    """
    Preprocess Cell Tracking Challenge (CTC) format files.

    Expects files in the standard CTC format, optionally with an additional column indicating right censoring.
    Columns are L B E P (R):
    L - a unique label of the tracked cell (any positive number, not zero or negative)
    B - the frame where the cell begins (integer)
    E - the frame where the cell ends (integer)
    P - label of the parent cell (0 is used when the parent is unknown)
    R - right censoring flag (1=right-censored, 0=not). Note a 0 only means it is not manually declared as
    right-censored. It will still be considered right-censored by the processing code, if the last observed frame
    coincides with the end of the movie.
    """
    tracks = pd.read_table(input_ctc_filepath, sep=r"\s+", header=None)
    validate_tracks_shape_dtypes(tracks)

    tracks = validate_parents_in_labels(tracks)
    tracks = validate_parent_label_unequal(tracks)

    if fix_late_daughters:
        tracks = correct_late_daughters(tracks)

    if fix_missing_daughters:
        tracks = correct_missing_daughters(tracks)

    tracks = validate_cell_begin_end_frames(tracks)
    tracks = validate_n_daughters(tracks)
    tracks = validate_mother_daughter_frames(tracks)
    tracks = validate_right_censored_daughters(tracks)

    # save new file
    tracks.to_csv(output_ctc_filepath, sep=" ", header=False, index=False)