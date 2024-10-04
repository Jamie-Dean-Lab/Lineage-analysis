import logging
from pathlib import Path

import pandas as pd
from btrack.io import HDF5FileHandler

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_btrack_file(input_btrack_filepath: Path) -> pd.DataFrame:
    """
    Preprocess btrack (.h5) format files.

    Expects files in the standard btrack (.h5) output format.
    """
    # Extract LBEP table from the btrack file
    with HDF5FileHandler(input_btrack_filepath, "r") as reader:
        tracks_lbep = reader.lbep

    # We only need the first four columns (LBEP) -
    # btrack has two additional for R (root track) and G (generational depth)
    ctc_table = pd.DataFrame(tracks_lbep[:, 0:4], columns=["L", "B", "E", "P"])

    logger.info("Extracted CTC table from btrack .h5")

    return ctc_table
