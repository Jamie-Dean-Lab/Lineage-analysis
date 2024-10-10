import logging
from pathlib import Path

import pandas as pd

from track_converter.src.utils import convert_to_ctc

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_trackmate_csv(csv_filepath: Path) -> pd.DataFrame:
    # First four rows of a trackmate csv are headers - keep first and discard rest
    return pd.read_csv(csv_filepath, skiprows=[1, 2, 3])


def preprocess_trackmate_file(spots_csv_filepath: Path, edges_csv_filepath: Path) -> pd.DataFrame:
    """Preprocess trackmate format files."""
    spots = _read_trackmate_csv(spots_csv_filepath)
    edges = _read_trackmate_csv(edges_csv_filepath)

    ctc_table = convert_to_ctc(spots, edges, "ID", "FRAME", "SPOT_SOURCE_ID", "SPOT_TARGET_ID", "TRACK_ID")

    # For this, need to know which cells (i.e. which row of the CTC table) each spot id belongs to

    # check if they have children - abort if so

    # mark as not right-censored

    logger.info("Extracted CTC table from trackmate files")

    return ctc_table
