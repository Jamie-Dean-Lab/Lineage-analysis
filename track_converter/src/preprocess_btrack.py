import logging
from pathlib import Path

import pandas as pd
from btrack.btypes import Tracklet
from btrack.constants import Fates
from btrack.io import HDF5FileHandler

from track_converter.src.utils import discard_all_descendants

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def right_censor_terminate_fates(tracks: list[Tracklet], ctc_table: pd.DataFrame) -> pd.DataFrame:
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


def discard_false_positives(tracks: list[Tracklet], ctc_table: pd.DataFrame) -> pd.DataFrame:
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


def preprocess_btrack_file(
    input_btrack_filepath: Path, use_terminate_fates: bool = True, remove_false_positives: bool = True
) -> pd.DataFrame:
    """
    Preprocess btrack (.h5) format files.

    Expects files in the standard btrack (.h5) output format.
    """
    # Extract LBEP table from the btrack file
    with HDF5FileHandler(input_btrack_filepath, "r") as reader:
        tracks = reader.tracks
        tracks_lbep = reader.lbep

    # We only need the first four columns (LBEP) -
    # btrack has two additional for R (root track) and G (generational depth)
    ctc_table = pd.DataFrame(tracks_lbep[:, 0:4], columns=["L", "B", "E", "P"])

    # btrack uses the cell label as the parent when it is unknown (rather than zero). Correct this to match CTC format.
    ctc_table.loc[ctc_table["L"] == ctc_table["P"], "P"] = 0

    if use_terminate_fates:
        ctc_table = right_censor_terminate_fates(tracks, ctc_table)

    if remove_false_positives:
        ctc_table = discard_false_positives(tracks, ctc_table)

    logger.info("Extracted CTC table from btrack .h5")

    return ctc_table
