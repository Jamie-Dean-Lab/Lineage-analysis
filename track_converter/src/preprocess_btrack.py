import logging
from pathlib import Path

import pandas as pd
from btrack.btypes import Tracklet
from btrack.constants import Fates
from btrack.io import HDF5FileHandler

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def right_censor_terminate_fates(tracks: list[Tracklet], ctc_table: pd.DataFrame) -> pd.DataFrame:
    """Mark any track with a fate of TERMINATE_BACK, TERMINATE_BORDER or TERMINATE_LAZY as right-censored."""
    terminate_fates = (Fates.TERMINATE, Fates.TERMINATE_BACK, Fates.TERMINATE_BORDER, Fates.TERMINATE_LAZY)

    # Add column for right-censoring (1 = right censored, 0 = not)
    ctc_table["R"] = 0

    for track in tracks:
        if track.fate in terminate_fates:
            ctc_table.loc[ctc_table.L == track.ID, "R"] = 1

    return ctc_table


def preprocess_btrack_file(input_btrack_filepath: Path, use_terminate_fates: bool = True) -> pd.DataFrame:
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

    if use_terminate_fates:
        ctc_table = right_censor_terminate_fates(tracks, ctc_table)

    logger.info("Extracted CTC table from btrack .h5")

    return ctc_table