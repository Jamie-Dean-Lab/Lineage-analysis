import logging
from pathlib import Path

import numpy as np
import pandas as pd

from track_converter.src.preprocess_ctc import preprocess_ctc_file
from track_converter.src.utils import check_dead_spots_have_no_children, convert_to_ctc

logger = logging.getLogger(__name__)


def _read_trackmate_csv(csv_filepath: Path) -> pd.DataFrame:
    # First four rows of a trackmate csv are headers - keep first and discard rest
    return pd.read_csv(csv_filepath, skiprows=[1, 2, 3])


def _read_dead_cell_labels(spots: pd.DataFrame, edges: pd.DataFrame, dead_label: str) -> np.array:
    """Read cell labels manually tagged as dead via the trackmate/mamut user interface."""
    dead_spots = spots.loc[dead_label == spots.LABEL, :]
    check_dead_spots_have_no_children(dead_spots, edges, "ID", "SPOT_SOURCE_ID")
    return dead_spots.ctc_label.unique()


def preprocess_trackmate_or_mamut_files(
    spots_csv_filepath: Path,
    edges_csv_filepath: Path,
    output_ctc_filepath: Path,
    fix_late_daughters: bool = False,
    fix_missing_daughters: bool = False,
    dead_label: str | None = None,
) -> None:
    """
    Preprocess TrackMate or MaMuT format csv files.

    Convert Trackmate / MaMuT csv files into the cell tracking challenge (CTC) format, with an additional column
    for right-censoring. TrackMate and MaMuT share the same output csv file formats and so can be processed in the
    same way. All cells that don't end in cell division will be marked as right-censored, except for (optionally)
    those manually labelled as dead in TrackMate or MaMuT.

    Parameters
    ----------
    spots_csv_filepath : Path
        Path to spots csv file.
    edges_csv_filepath : Path
        Path to edges csv file.
    output_ctc_filepath : Path
        Path to save output .txt file.
    fix_late_daughters : bool, optional
        Whether to back-date any late daughters to the start time of the earlier daughter.
    fix_missing_daughters : bool, optional
        Whether to create a second daughter for any mother cells that only have one.
    dead_label : str | None, optional
        Name of manually labelled 'dead' spots. If provided, these will be marked as not right-censored.

    """
    spots = _read_trackmate_csv(spots_csv_filepath)
    edges = _read_trackmate_csv(edges_csv_filepath)

    ctc_table, spots = convert_to_ctc(spots, edges, "ID", "FRAME", "SPOT_SOURCE_ID", "SPOT_TARGET_ID", "TRACK_ID")

    if dead_label is not None:
        dead_ctc_labels = _read_dead_cell_labels(spots, edges, dead_label)
    else:
        dead_ctc_labels = []

    logger.info("Extracted CTC table from trackmate/mamut files")

    preprocess_ctc_file(
        ctc_table,
        output_ctc_filepath,
        fix_late_daughters=fix_late_daughters,
        fix_missing_daughters=fix_missing_daughters,
        default_right_censor=True,
        dead_cell_labels=dead_ctc_labels,
    )
