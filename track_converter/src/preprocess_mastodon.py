import logging
from pathlib import Path

import numpy as np
import pandas as pd

from track_converter.src.preprocess_ctc import preprocess_ctc_file
from track_converter.src.utils import convert_to_ctc

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_mastodon_csv(csv_filepath: Path) -> pd.DataFrame:
    # First three rows of a mastodon csv are headers - the first two contain information we need, can drop the third.
    mastodon_df = pd.read_csv(csv_filepath, header=[0, 1], skiprows=[2])

    # For ease of use, combine both headers into one
    combined_header_names = []
    for header_1, header_2 in mastodon_df.columns:
        if (header_2 in ("", " ")) or (header_2.startswith("Unnamed")):
            combined_header_names.append(header_1)
        else:
            combined_header_names.append(f"{header_1}-{header_2}")
    mastodon_df.columns = combined_header_names

    return mastodon_df


def _read_dead_cell_labels(spots: pd.DataFrame, links: pd.DataFrame, dead_tagset: str, dead_tag: str) -> np.array:
    dead_spots = spots.loc[spots[f"{dead_tagset}-{dead_tag}"] == 1, :]

    # Check spots have no children (they should be at the very end of a cell track)
    spots_are_invalid = dead_spots.ID.isin(links["Link target IDs-Source spot id"])
    if (spots_are_invalid).any():
        invalid_spots = dead_spots.ID[spots_are_invalid].to_numpy()
        msg = (
            f"Spots with ids {invalid_spots} that were tagged as dead have child spots. These spots should lie "
            f"at the very end of a track."
        )
        logger.error(msg)
        raise ValueError(msg)

    return dead_spots.ctc_label.unique()


def preprocess_mastodon_file(
    spots_csv_filepath: Path,
    links_csv_filepath: Path,
    output_ctc_filepath: Path,
    dead_tagset: str | None = None,
    dead_tag: str | None = None,
) -> None:
    """Preprocess mastodon format files."""
    spots = _read_mastodon_csv(spots_csv_filepath)
    links = _read_mastodon_csv(links_csv_filepath)

    ctc_table, spots = convert_to_ctc(
        spots, links, "ID", "Spot frame", "Link target IDs-Source spot id", "Link target IDs-Target spot id"
    )

    if (dead_tagset is not None) and (dead_tag is not None):
        dead_ctc_labels = _read_dead_cell_labels(spots, links, dead_tagset, dead_tag)
    else:
        dead_ctc_labels = []

    logger.info("Extracted CTC table from mastodon files")

    preprocess_ctc_file(ctc_table, output_ctc_filepath, default_right_censor=True, dead_cell_labels=dead_ctc_labels)
