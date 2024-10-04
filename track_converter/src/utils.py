import logging

import pandas as pd

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _find_root(tracks: pd.DataFrame, cell_label: int) -> int:
    """
    Find the root of the tree containing the given cell_label (assumes tracks in CTC format).

    i.e. follow the parents up until you find a parent that is zero.
    """
    parent = tracks.loc[cell_label == tracks.L, "P"].to_numpy()[0]

    if parent == 0:
        return cell_label

    return _find_root(tracks, parent)


def discard_all_descendants(tracks: pd.DataFrame, cell_label: int) -> pd.DataFrame:
    """Remove all descendants of the given cell_label (assumes tracks in CTC format)."""
    children = tracks.loc[cell_label == tracks.P, "L"]
    for child_label in children.to_numpy():
        # Remove the child label
        tracks = tracks.drop(tracks[child_label == tracks.L].index)
        # Remove its children
        tracks = discard_all_descendants(tracks, child_label)

    return tracks


def discard_related_cells(tracks: pd.DataFrame, cell_labels: list[int]) -> pd.DataFrame:
    """
    Discard all tracked cells that are related to the given cell labels (assumes tracks in CTC format).

    This includes the cell itself, as well as all descendants, ancestors and siblings i.e. any cells in the tree
    connected to them.
    """
    for label in cell_labels:
        if label in tracks["L"].to_numpy():
            root = _find_root(tracks, label)
            tracks = discard_all_descendants(tracks, root)  # discard all descendants of the root
            tracks = tracks.drop(tracks[root == tracks.L].index)  # discard the root itself

    if tracks.empty:
        msg = f"No tracks remaining after discarding related cells of {cell_labels}"
        logger.error(msg)
        raise ValueError(msg)

    return tracks
