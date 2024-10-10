import logging
from pathlib import Path

import pandas as pd

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


def preprocess_mastodon_file(spots_csv_filepath: Path, links_csv_filepath: Path) -> pd.DataFrame:
    """Preprocess mastodon format files."""
    spots = _read_mastodon_csv(spots_csv_filepath)
    links = _read_mastodon_csv(links_csv_filepath)

    ctc_table = convert_to_ctc(
        spots, links, "ID", "Spot frame", "Link target IDs-Source spot id", "Link target IDs-Target spot id"
    )

    # Get spots tagged as dead

    # Check they have no children

    logger.info("Extracted CTC table from mastodon files")

    return ctc_table
