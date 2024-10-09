import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logging.basicConfig(format="%(levelname)s: %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_trackmate_csv(csv_filepath: Path) -> pd.DataFrame:
    # First four rows of a trackmate csv are headers - keep first and discard rest
    return pd.read_csv(csv_filepath, skiprows=[1, 2, 3])


def _convert_to_ctc(spots: pd.DataFrame, edges: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Convert Trackmate spots/edges/tracks tables to CTC format."""
    # don't allow any merges between cells
    tracks_are_invalid = tracks.NUMBER_MERGES > 0
    if (tracks_are_invalid).any():
        invalid_track_ids = tracks.loc[tracks_are_invalid, "TRACK_ID"]
        msg = (
            f"Tracks with ids {invalid_track_ids} have merges between cells - "
            f"this isn't allowed in cell tracking challenge format files"
        )
        logger.error(msg)
        raise ValueError(msg)

    # Add column to spots to keep track of which ctc_label they are assigned to
    spots["ctc_label"] = 0

    # Add column to keep track of the parent ctc label (for the first spot in each cell track)
    spots["parent_ctc_label"] = 0

    # Construct a graph of each Trackmate track (i.e. a cell and all of its descendants) and split it into individual
    # cell tracks (each with its own ctc_label)
    ctc_label = 1
    for track_id in spots.TRACK_ID.unique():
        track_spots = spots.loc[track_id == spots.TRACK_ID, :]
        track_edges = edges.loc[track_id == edges.TRACK_ID, :]

        track_graph = nx.DiGraph()
        track_graph.add_nodes_from(track_spots.ID)
        track_graph.add_edges_from(list(zip(track_edges.SPOT_SOURCE_ID, track_edges.SPOT_TARGET_ID, strict=False)))

        if not nx.is_directed_acyclic_graph(track_graph):
            msg = f"Graph of track id {track_id} contains cycles - tracks should be acyclic"
            logger.error(msg)
            raise ValueError(msg)

        if not nx.is_arborescence(track_graph):
            msg = (
                f"Graph of track id {track_id} isn't an 'arborescence' i.e. it isn't a connected tree with each node "
                f"having at most one parent. Does your track contain merges between cells?"
            )
            logger.error(msg)
            raise ValueError(msg)

        root = [node for node, degree in track_graph.in_degree() if degree == 0]
        if len(root) != 1:
            msg = f"Couldn't find root of track id {track_id}"
            logger.error(msg)
            raise ValueError(msg)

        # loop through nodes via depth first search from the root node
        for node in nx.dfs_preorder_nodes(track_graph, source=root[0]):
            node_out_degree = track_graph.out_degree[node]
            if node_out_degree > 2:
                msg = f"Spot with id {node} has more than 2 children."
                logger.error(msg)
                raise ValueError(msg)

            spots.loc[node == spots.ID, "ctc_label"] = ctc_label

            if node_out_degree == 2:
                # this is a branch point, and therefore the last node in this cell track
                # mark the child nodes with the correct parent_ctc_label
                for child_node in track_graph.successors(node):
                    spots.loc[child_node == spots.ID, "parent_ctc_label"] = ctc_label

                ctc_label += 1
            elif node_out_degree == 0:
                # this is a leaf of the graph, and therefore the last node in this cell track
                ctc_label += 1

    # Construct CTC table
    ctc_labels = spots["ctc_label"].unique()
    ctc_labels.sort()

    ctc_columns = {
        "L": ctc_labels,
        "B": spots.groupby("ctc_label", sort=True).FRAME.min(),
        "E": spots.groupby("ctc_label", sort=True).FRAME.max(),
        "P": spots.groupby("ctc_label", sort=True).parent_ctc_label.max(),
    }
    ctc_table = pd.DataFrame(data=ctc_columns)

    return ctc_table.reset_index(drop=True)


def preprocess_trackmate_file(
    spots_csv_filepath: Path, edges_csv_filepath: Path, tracks_csv_filepath: Path
) -> pd.DataFrame:
    """Preprocess trackmate format files."""
    spots = _read_trackmate_csv(spots_csv_filepath)
    edges = _read_trackmate_csv(edges_csv_filepath)
    tracks = _read_trackmate_csv(tracks_csv_filepath)

    ctc_table = _convert_to_ctc(spots, edges, tracks)

    # For this, need to know which cells (i.e. which row of the CTC table) each spot id belongs to

    # check if they have children - abort if so

    # mark as not right-censored

    logger.info("Extracted CTC table from trackmate files")

    return ctc_table
