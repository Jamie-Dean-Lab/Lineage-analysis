from pandas.testing import assert_frame_equal

from track_converter.src.preprocess_mastodon import preprocess_mastodon_file
from track_converter.tests.utils import read_tracks


def test_preprocess_trackmate(mastodon_test_data_dir):
    """Run a valid example file through the pre-processing and check it matches the expected CTC table."""
    spots = mastodon_test_data_dir / "MastodonTable-Spot.csv"
    links = mastodon_test_data_dir / "MastodonTable-Link.csv"
    ctc_table = preprocess_mastodon_file(spots, links)

    expected_ctc_table = read_tracks(mastodon_test_data_dir / "expected_ctc_output.txt")

    assert_frame_equal(ctc_table, expected_ctc_table)
