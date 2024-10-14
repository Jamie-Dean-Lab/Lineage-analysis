import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from btrack.btypes import Tracklet
from btrack.constants import Fates
from btrack.io import HDF5FileHandler

from track_converter.src.preprocess_ctc import preprocess_ctc_file
from track_converter.src.utils import discard_all_descendants

logger = logging.getLogger(__name__)


def _right_censor_terminate_fates(tracks: list[Tracklet], ctc_table: pd.DataFrame) -> pd.DataFrame:
    """Mark any track with a fate of TERMINATE_BACK, TERMINATE_BORDER or TERMINATE_LAZY as right-censored."""
    logger.info("Marking cells with btrack TERMINATE fates as right-censored")
    terminate_fates = (Fates.TERMINATE, Fates.TERMINATE_BACK, Fates.TERMINATE_BORDER, Fates.TERMINATE_LAZY)

    # Add column for right-censoring (1 = right censored, 0 = not)
    ctc_table["R"] = 0

    for track in tracks:
        if (track.fate in terminate_fates) and (track.ID in ctc_table.L.to_numpy()):
            ctc_table.loc[ctc_table.L == track.ID, "R"] = 1

            # If a track with terminate fate has children, remove them
            if (ctc_table["P"] == track.ID).any():
                msg = f"Track with id {track.ID} has a TERMINATE fate, but has children. Removing children..."
                logger.warning(msg)
                ctc_table = discard_all_descendants(ctc_table, track.ID)

    return ctc_table


def _discard_false_positives(tracks: list[Tracklet], ctc_table: pd.DataFrame) -> pd.DataFrame:
    """Remove any cells with a fate of FALSE_POSITIVE, as well as all of their descendants (if any)."""
    logger.info("Removing cells with btrack FALSE_POSITIVE fate")

    for track in tracks:
        if (track.fate == Fates.FALSE_POSITIVE) and (track.ID in ctc_table.L.to_numpy()):
            # Remove the false positive track
            ctc_table = ctc_table.drop(ctc_table[track.ID == ctc_table.L].index)

            # If a track with false positive fate has children, remove them
            if (ctc_table["P"] == track.ID).any():
                msg = f"Track with id {track.ID} has a FALSE_POSITIVE fate, but has children. Removing children..."
                logger.warning(msg)
                ctc_table = discard_all_descendants(ctc_table, track.ID)

    return ctc_table


def _validate_dead_track_ids(ctc_table: pd.DataFrame, dead_track_ids: Iterable[int]) -> None:
    # check all dead_track_ids appear in the table (L)
    dead_ids_series = pd.Series(dead_track_ids)
    ids_are_valid = dead_ids_series.isin(ctc_table.L)
    if (~ids_are_valid).any():
        invalid_ids = dead_ids_series[~ids_are_valid].to_numpy()
        msg = f"dead_track_ids {invalid_ids} don't match any track IDs"
        logger.error(msg)
        raise ValueError(msg)

    # Check all dead_track_ids have no children
    ids_are_invalid = dead_ids_series.isin(ctc_table.P)
    if (ids_are_invalid).any():
        invalid_ids = dead_ids_series[ids_are_invalid].to_numpy()
        msg = (
            f"Tracks with IDs {invalid_ids} were provided as dead_track_ids, but have children. "
            f"These tracks should end with cell death, not division."
        )
        logger.error(msg)
        raise ValueError(msg)


def preprocess_btrack_file(
    btrack_h5_filepath: Path,
    output_ctc_filepath: Path,
    use_terminate_fates: bool = True,
    remove_false_positives: bool = True,
    fix_late_daughters: bool = False,
    fix_missing_daughters: bool = False,
    dead_track_ids: Iterable[int] | None = None,
) -> None:
    """
    Preprocess btrack format (.h5) file.

    Convert a btrack output file (.h5) into the cell tracking challenge (CTC) format, with an additional column
    for right-censoring. If use_terminate_fates is True, then all right_censoring information will be read directly
    from btrack's assigned 'fates'. If False, then all cells that don't end in cell division will be marked as
    right-censored except for (optionally) those provided under dead_track_ids.

    Parameters
    ----------
    btrack_h5_filepath : Path
        Path to btrack output .h5 file.
    output_ctc_filepath : Path
        Path to save output .txt file.
    use_terminate_fates : bool
        Mark any track with a btrack fate of TERMINATE_BACK, TERMINATE_BORDER or TERMINATE_LAZY as right-censored.
        If this is True, any dead_track_ids will be ignored.
    remove_false_positives : bool
        Remove any cells with a btrack fate of FALSE_POSITIVE, as well as all of their descendants (if any).
    fix_late_daughters : bool, optional
        Whether to back-date any late daughters to the start time of the earlier daughter.
    fix_missing_daughters : bool, optional
        Whether to create a second daughter for any mother cells that only have one.
    dead_track_ids : Iterable[int] | None
        List of track ids (accessed via .ID for each btrack Tracklet) to consider as 'dead' cells. These will be
        marked as not right-censored.

    """
    # Extract LBEP table from the btrack file
    with HDF5FileHandler(btrack_h5_filepath, "r") as reader:
        tracks = reader.tracks
        tracks_lbep = reader.lbep

    # We only need the first four columns (LBEP) -
    # btrack has two additional for R (root track) and G (generational depth)
    ctc_table = pd.DataFrame(tracks_lbep[:, 0:4], columns=["L", "B", "E", "P"])

    # btrack uses the cell label as the parent when it is unknown (rather than zero). Correct this to match CTC format.
    ctc_table.loc[ctc_table["L"] == ctc_table["P"], "P"] = 0

    if use_terminate_fates:
        ctc_table = _right_censor_terminate_fates(tracks, ctc_table)

    if remove_false_positives:
        ctc_table = _discard_false_positives(tracks, ctc_table)

    logger.info("Extracted CTC table from btrack .h5")

    if use_terminate_fates:
        preprocess_ctc_file(
            ctc_table,
            output_ctc_filepath,
            fix_late_daughters=fix_late_daughters,
            fix_missing_daughters=fix_missing_daughters,
            default_right_censor=False,
        )
        return

    if dead_track_ids is not None:
        _validate_dead_track_ids(ctc_table, dead_track_ids)
        dead_ctc_labels = dead_track_ids
    else:
        dead_ctc_labels = []

    preprocess_ctc_file(
        ctc_table,
        output_ctc_filepath,
        fix_late_daughters=fix_late_daughters,
        fix_missing_daughters=fix_missing_daughters,
        default_right_censor=True,
        dead_cell_labels=dead_ctc_labels,
    )
