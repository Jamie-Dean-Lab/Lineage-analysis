import logging

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def _find_ctc_root(tracks: pd.DataFrame, cell_label: int) -> int:
    """
    Find the root of the tree containing the given cell_label (assumes tracks in CTC format).

    i.e. follow the parents up until you find a parent that is zero.
    """
    parent = tracks.loc[cell_label == tracks.L, "P"].to_numpy()[0]

    if parent == 0:
        return cell_label

    return _find_ctc_root(tracks, parent)


def discard_all_descendants(tracks: pd.DataFrame, cell_label: int) -> pd.DataFrame:
    """
    Remove all descendants of the given cell_label.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks in CTC format (LBEP)
    cell_label : int
        Cell label (L) to remove descendants of.

    Returns
    -------
    pd.DataFrame
        CTC format table with the cell's descendants removed.

    """
    children = tracks.loc[cell_label == tracks.P, "L"]
    for child_label in children.to_numpy():
        # Remove the child label
        tracks = tracks.drop(tracks[child_label == tracks.L].index)
        # Remove its children
        tracks = discard_all_descendants(tracks, child_label)

    return tracks


def discard_related_cells(tracks: pd.DataFrame, cell_labels: list[int]) -> pd.DataFrame:
    """
    Discard all tracked cells that are related to the given cell labels.

    This includes the cell itself, as well as all descendants, ancestors and siblings i.e. any cells in the tree
    connected to them.

    Parameters
    ----------
    tracks : pd.DataFrame
        Tracks in CTC format (LBEP)
    cell_labels : list[int]
        Cell labels (L) to remove all related cells for

    Returns
    -------
    pd.DataFrame
        CTC format table with cells removed

    """
    for label in cell_labels:
        if label in tracks["L"].to_numpy():
            root = _find_ctc_root(tracks, label)
            tracks = discard_all_descendants(tracks, root)  # discard all descendants of the root
            tracks = tracks.drop(tracks[root == tracks.L].index)  # discard the root itself

    if tracks.empty:
        msg = f"No tracks remaining after discarding related cells of {cell_labels}"
        logger.error(msg)
        raise ValueError(msg)

    return tracks


def check_dead_spots_have_no_children(
    dead_spots: pd.DataFrame, edges: pd.DataFrame, spot_id_col: str, edge_source_id_col: str
) -> None:
    """
    Throw an error if any of the dead spots have children.

    Parameters
    ----------
    dead_spots : pd.DataFrame
        Spot table (from trackmate / mamut / mastodon) filtered to only contain dead spots.
    edges : pd.DataFrame
        Edges table (form trackmate / mamut / mastodon)
    spot_id_col : str
        Name of column in spot table that contains the ID of each spot
    edge_source_id_col : str
        Name of column in edges table that contains the ID of the source spot

    """
    spots_are_invalid = dead_spots[spot_id_col].isin(edges[edge_source_id_col])
    if (spots_are_invalid).any():
        invalid_spots = dead_spots[spot_id_col][spots_are_invalid].to_numpy()
        msg = (
            f"Spots with ids {invalid_spots} that were tagged as dead have child spots. These spots should lie "
            f"at the very end of a track."
        )
        logger.error(msg)
        raise ValueError(msg)


def convert_to_ctc(
    spots: pd.DataFrame,
    edges: pd.DataFrame,
    spot_id_col: str,
    spot_frame_col: str,
    edge_source_id_col: str,
    edge_target_id_col: str,
    track_id_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert tables of spots/edges information into the CTC format.

    This will work for TrackMate, MaMuT or Mastodon csv files.

    Parameters
    ----------
    spots : pd.DataFrame
        DataFrame of all detected spots (one per row)
    edges : pd.DataFrame
        DataFrame of edges linking two spots together (one edge per row)
    spot_id_col : str
        Name of column in spot table that contains the ID of each spot
    spot_frame_col : str
        Name of column in spots table that contains the frame number
    edge_source_id_col : str
        Name of column in edges table that contains the ID of the source spot
    edge_target_id_col : str
        Name of column in edges table that contains the ID of the target spot
    track_id_col : str, optional
        Name of column in spots table that contains the track ID each belongs to. If available, this is used
        for logging to print which track id any errors are found in.

    Returns
    -------
    ctc_table : pd.DataFrame
        Tracks converted to CTC format i.e. 4 columns (L B E P)
    spots : pd.DataFrame
        The input spots DataFrame with an added column 'ctc_label' indicating which CTC label (L) each spot belongs to

    """
    # Add column to spots to keep track of which ctc_label they are assigned to
    spots["ctc_label"] = 0

    # Add column to keep track of the parent ctc label (for the first spot in each cell track)
    spots["parent_ctc_label"] = 0

    # Construct a graph of all tracks, then split each into individual cell tracks (each with its own ctc_label)
    ctc_label = 1
    tracks_graph = nx.DiGraph()
    tracks_graph.add_nodes_from(spots[spot_id_col])
    tracks_graph.add_edges_from(list(zip(edges[edge_source_id_col], edges[edge_target_id_col], strict=False)))

    # Each weakly connected component is one 'track' i.e. one cell and all of its descendants
    for track_nodes in nx.weakly_connected_components(tracks_graph):
        track_graph = tracks_graph.subgraph(track_nodes)

        # Grab an arbitrary node in the track graph and look up the track id (if it exists)
        if track_id_col is not None:
            for node in track_graph:
                track_id = spots.loc[spots[spot_id_col] == node, track_id_col].to_numpy()[0]
                break
        else:
            track_id = None

        if not nx.is_directed_acyclic_graph(track_graph):
            msg = (
                "Graph "
                + ("of track id {track_id} " if track_id else "")
                + "contains cycles - tracks should be acyclic"
            )
            logger.error(msg)
            raise ValueError(msg)

        if not nx.is_arborescence(track_graph):
            msg = (
                "Graph "
                + ("of track id {track_id} " if track_id else "")
                + "isn't an 'arborescence' i.e. it isn't a connected tree with each node having at most one parent. "
                + "Does your track contain merges between cells?"
            )
            logger.error(msg)
            raise ValueError(msg)

        root = [node for node, degree in track_graph.in_degree() if degree == 0]
        if len(root) != 1:
            msg = "Couldn't find root of track" + (" with id {track_id}" if track_id else "")
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
        "B": spots.groupby("ctc_label", sort=True)[spot_frame_col].min(),
        "E": spots.groupby("ctc_label", sort=True)[spot_frame_col].max(),
        "P": spots.groupby("ctc_label", sort=True).parent_ctc_label.max(),
    }
    ctc_table = pd.DataFrame(data=ctc_columns)
    ctc_table = ctc_table.reset_index(drop=True)

    return ctc_table, spots.drop(["parent_ctc_label"], axis=1)
