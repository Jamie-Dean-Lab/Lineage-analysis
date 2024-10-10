import pandas as pd
import pytest
from btrack.constants import Fates
from btrack.io import HDF5FileHandler

from track_converter.src.preprocess_btrack import preprocess_btrack_file
from track_converter.tests.utils import read_tracks


@pytest.fixture
def invalid_cell_labels():
    # The btrack test data has one daughter cell that appears 2 frames after the mother, rather than one. Therefore,
    # the following cell labels (that are connected to it) are invalid
    return (185, 177, 184)


def test_btrack_using_terminate_fates(invalid_cell_labels, btrack_test_data_dir, tmp_path):
    """
    Run a valid example file through the pre-processing with use_terminate_fates on.

    This checks all parents / start / stop frames etc are set correctly, as well as all cells with terminate fates
    being assigned as right-censored.
    """
    tracks_in_path = btrack_test_data_dir / "tracks.h5"
    tracks_out_path = tmp_path / "tracks_out.txt"

    preprocess_btrack_file(tracks_in_path, tracks_out_path, use_terminate_fates=True, remove_false_positives=False)
    ctc_table = read_tracks(tracks_out_path)

    # Read tracks directly from file and check each appears properly in the CTC table
    with HDF5FileHandler(tracks_in_path, "r") as reader:
        tracks = reader.tracks

    assert (len(tracks) - len(invalid_cell_labels)) == ctc_table.shape[0]
    for track in tracks:
        if track.ID not in invalid_cell_labels:
            # valid cells should match btrack tracks
            parent = track.parent if track.parent != track.ID else 0

            if track.fate in (Fates.TERMINATE, Fates.TERMINATE_BACK, Fates.TERMINATE_BORDER, Fates.TERMINATE_LAZY):
                right_censor = 1
            else:
                right_censor = 0

            expected_row = pd.Series(
                (track.ID, track.start, track.stop, parent, right_censor), index=("L", "B", "E", "P", "R")
            )
            actual_row = ctc_table.loc[ctc_table.L == track.ID, :].squeeze()

            assert actual_row.equals(expected_row)
        else:
            # invalid cells should have been removed
            assert track.ID not in ctc_table.L.to_numpy()


def test_btrack_remove_false_positives(invalid_cell_labels, btrack_test_data_dir, tmp_path):
    """
    Run a valid example file through the pre-processing with remove_false_positives on.

    This checks all false positive cells are removed.
    """
    tracks_in_path = btrack_test_data_dir / "tracks.h5"
    tracks_out_path = tmp_path / "tracks_out.txt"

    preprocess_btrack_file(tracks_in_path, tracks_out_path, use_terminate_fates=True, remove_false_positives=True)
    ctc_table = read_tracks(tracks_out_path)
    cell_labels = ctc_table.L.to_numpy()

    # Read tracks directly from file and check all tracks are present, except for false positives
    with HDF5FileHandler(tracks_in_path, "r") as reader:
        tracks = reader.tracks

    assert len(tracks) != ctc_table.shape[0]
    for track in tracks:
        if track.fate == Fates.FALSE_POSITIVE:
            assert track.ID not in cell_labels
        elif track.ID not in invalid_cell_labels:
            assert track.ID in cell_labels
        else:
            assert track.ID not in cell_labels
