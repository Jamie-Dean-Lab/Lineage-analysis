import pandas as pd
from btrack.constants import Fates
from btrack.io import HDF5FileHandler

from track_converter.src.preprocess_btrack import preprocess_btrack_file


def test_valid_file_passes(btrack_test_data_dir):
    """Run a larger valid example file through the validation, checking no exceptions are raised."""
    tracks_in_path = btrack_test_data_dir / "tracks.h5"
    ctc_table = preprocess_btrack_file(tracks_in_path)

    # Read tracks directly from file and check each appears properly in the CTC table
    with HDF5FileHandler(tracks_in_path, "r") as reader:
        tracks = reader.tracks

    assert len(tracks) == ctc_table.shape[0]
    for track in tracks:
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
