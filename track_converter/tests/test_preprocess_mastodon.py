from pandas.testing import assert_frame_equal

from track_converter.src.preprocess_mastodon import preprocess_mastodon_files
from track_converter.tests.utils import read_tracks


def test_preprocess_mastodon(mastodon_test_data_dir, tmp_path):
    """Run a valid example file through the pre-processing and check it matches the expected CTC table."""
    spots = mastodon_test_data_dir / "MastodonTable-Spot.csv"
    links = mastodon_test_data_dir / "MastodonTable-Link.csv"
    tracks_out_path = tmp_path / "tracks_out.txt"

    preprocess_mastodon_files(spots, links, tracks_out_path, dead_tagset="dead_cells", dead_tag="dead")
    ctc_table = read_tracks(tracks_out_path)
    expected_ctc_table = read_tracks(mastodon_test_data_dir / "expected_ctc_output.txt")

    assert_frame_equal(ctc_table, expected_ctc_table)
