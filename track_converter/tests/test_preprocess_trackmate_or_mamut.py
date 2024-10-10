from pandas.testing import assert_frame_equal

from track_converter.src.preprocess_trackmate_or_mamut import preprocess_trackmate_or_mamut_files
from track_converter.tests.utils import read_tracks


def test_preprocess_trackmate_or_mamut(trackmate_test_data_dir, tmp_path):
    """Run a valid example file through the pre-processing and check it matches the expected CTC table."""
    spots = trackmate_test_data_dir / "FakeTracks_spots.csv"
    edges = trackmate_test_data_dir / "FakeTracks_edges.csv"
    tracks_out_path = tmp_path / "tracks_out.txt"

    preprocess_trackmate_or_mamut_files(spots, edges, tracks_out_path, dead_label="dead")
    ctc_table = read_tracks(tracks_out_path)
    expected_ctc_table = read_tracks(trackmate_test_data_dir / "expected_ctc_output.txt")

    assert_frame_equal(ctc_table, expected_ctc_table)
