import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def find_root(tracks: pd.DataFrame, cell_label: int) -> int:
    """
    Find the root of the tree containing the given cell_label.

    i.e. follow the parents up until you find a parent that is zero.
    """
    parent = tracks.loc[cell_label == tracks.L, "P"].to_numpy()[0]

    if parent == 0:
        return cell_label

    return find_root(tracks, parent)


def discard_all_descendants(tracks: pd.DataFrame, cell_label: int) -> pd.DataFrame:
    """Remove all descendants of the given cell_label, as well as itself."""
    # Remove its children
    children = tracks.loc[cell_label == tracks.P, "L"]
    for child_label in children.to_numpy():
        tracks = discard_all_descendants(tracks, child_label)

    # Remove the given cell label
    return tracks.drop(tracks[cell_label == tracks.L].index)


def discard_related_cells(tracks: pd.DataFrame, cell_labels: list[int]) -> pd.DataFrame:
    """
    Discard all tracked cells that are related to the given cell labels.

    This includes the cell itself, as well as all descendants, ancestors and siblings i.e. any cells in the tree
    connected to them.
    """
    for label in cell_labels:
        if label in tracks["L"].to_numpy():
            root = find_root(tracks, label)
            tracks = discard_all_descendants(tracks, root)

    if tracks.empty:
        msg = f"No tracks remaining after discarding related cells of {cell_labels}"
        logger.error(msg)
        raise ValueError(msg)

    return tracks


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


def validate_mother_daughter_frames(tracks: pd.DataFrame) -> pd.DataFrame:
    """Validate that both daughter cells appear one frame after the mother's last frame."""
    # ignore parent ids of 0 (this means the parent is unknown)
    tracks_with_parents = tracks.loc[tracks["P"] != 0, :]

    # For each tracked cell, its begin frame (B) must be one more than its parent's end frame (E)
    tracks_with_parents = tracks_with_parents.merge(
        tracks[["L", "E"]], how="left", left_on="P", right_on="L", suffixes=("_track", "_parent")
    )
    tracks_with_parents = tracks_with_parents[["L_track", "P", "B", "E_parent"]]

    cells_are_invalid = tracks_with_parents["B"] - tracks_with_parents["E_parent"] != 1
    if cells_are_invalid.any():
        invalid_labels = tracks_with_parents.loc[cells_are_invalid, "L_track"]
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


def preprocess_ctc_file(input_ctc_filepath: Path, output_ctc_filepath: Path) -> None:
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
    tracks = validate_cell_begin_end_frames(tracks)
    tracks = validate_parents_in_labels(tracks)
    tracks = validate_parent_label_unequal(tracks)
    tracks = validate_n_daughters(tracks)
    tracks = validate_mother_daughter_frames(tracks)
    tracks = validate_right_censored_daughters(tracks)

    # save new file
    tracks.to_csv(output_ctc_filepath, sep=" ", header=False, index=False)
