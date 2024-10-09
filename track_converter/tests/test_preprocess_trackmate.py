from pandas.testing import assert_frame_equal

from track_converter.src.preprocess_trackmate import preprocess_trackmate_file
from track_converter.tests.utils import read_tracks


def test_preprocess_trackmate(trackmate_test_data_dir):
    """Run a valid example file through the pre-processing and check it matches the expected CTC table."""
    spots = trackmate_test_data_dir / "FakeTracks_spots.csv"
    edges = trackmate_test_data_dir / "FakeTracks_edges.csv"
    tracks = trackmate_test_data_dir / "FakeTracks_tracks.csv"
    ctc_table = preprocess_trackmate_file(spots, edges, tracks)

    expected_ctc_table = read_tracks(trackmate_test_data_dir / "expected_ctc_output.txt")

    assert_frame_equal(ctc_table, expected_ctc_table)
