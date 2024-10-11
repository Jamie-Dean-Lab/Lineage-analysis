import logging
from pathlib import Path

import numpy as np
import pandas as pd

from track_converter.src.preprocess_ctc import preprocess_ctc_file
from track_converter.src.utils import check_dead_spots_have_no_children, convert_to_ctc

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
    check_dead_spots_have_no_children(dead_spots, links, "ID", "Link target IDs-Source spot id")
    return dead_spots.ctc_label.unique()


def preprocess_mastodon_files(
    spots_csv_filepath: Path,
    links_csv_filepath: Path,
    output_ctc_filepath: Path,
    fix_late_daughters: bool = False,
    fix_missing_daughters: bool = False,
    dead_tagset: str | None = None,
    dead_tag: str | None = None,
) -> None:
    """
    Preprocess Mastodon format csv files.

    Convert Mastodon csv files into the cell tracking challenge (CTC) format, with an additional column
    for right-censoring. All cells that don't end in cell division will be marked as right-censored,
    except for (optionally) those manually tagged as dead in Mastodon.

    Parameters
    ----------
    spots_csv_filepath : Path
        Path to spots csv file.
    links_csv_filepath : Path
        Path to links csv file.
    output_ctc_filepath : Path
        Path to save output .txt file.
    fix_late_daughters : bool, optional
        Whether to back-date any late daughters to the start time of the earlier daughter.
    fix_missing_daughters : bool, optional
        Whether to create a second daughter for any mother cells that only have one.
    dead_tagset : str | None, optional
        Name of tagset for manually labelled 'dead' spots (dead_tag must also be provided)
    dead_tag : str | None, optional
        Name of tag (inside dead_tagset) for labelled 'dead' spots (dead_tagset must also be provided).
        All tagged cells will be marked as not right-censored.

    """
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

    preprocess_ctc_file(
        ctc_table,
        output_ctc_filepath,
        fix_late_daughters=fix_late_daughters,
        fix_missing_daughters=fix_missing_daughters,
        default_right_censor=True,
        dead_cell_labels=dead_ctc_labels,
    )
