from unittest import mock

import pytest
from click.testing import CliRunner

from track_converter.src.convert_tracks import convert_tracks


def test_btrack_cli(btrack_test_data_dir, tracks_out_path):
    """Test CLI call to btrack calls preprocess_btrack_file with correct parameters."""
    runner = CliRunner()

    with mock.patch(
        "track_converter.src.convert_tracks.preprocess_btrack_file", autospec=True
    ) as preprocess_btrack_file:
        input_path = btrack_test_data_dir / "tracks.h5"
        input_path_str = str(input_path.resolve())
        output_path_str = str(tracks_out_path.resolve())

        result = runner.invoke(
            convert_tracks,
            [
                "btrack",
                input_path_str,
                output_path_str,
                "--no-terminate-fates",
                "--keep-false-positives",
                "--fix-late-daughters",
                "--fix-missing-daughters",
                "--dead-track-ids",
                "[53, 49, 38]",
                "-v",
            ],
        )

        assert result.exit_code == 0
        preprocess_btrack_file.assert_called_once_with(
            input_path,
            tracks_out_path,
            use_terminate_fates=False,
            remove_false_positives=False,
            fix_late_daughters=True,
            fix_missing_daughters=True,
            dead_track_ids=[53, 49, 38],
        )


@pytest.mark.parametrize("tracking_software", ["trackmate", "mamut"])
def test_trackmate_mamut_cli(tracking_software, trackmate_test_data_dir, tracks_out_path):
    """Test CLI call to trackmate/mamut calls preprocess_trackmate_or_mamut_files with correct parameters."""
    runner = CliRunner()

    with mock.patch(
        "track_converter.src.convert_tracks.preprocess_trackmate_or_mamut_files", autospec=True
    ) as preprocess_trackmate_or_mamut_files:
        input_spot_path = trackmate_test_data_dir / "FakeTracks_spots.csv"
        input_edge_path = trackmate_test_data_dir / "FakeTracks_edges.csv"
        input_spot_str = str(input_spot_path.resolve())
        input_edge_str = str(input_edge_path.resolve())
        output_path_str = str(tracks_out_path.resolve())

        result = runner.invoke(
            convert_tracks,
            [
                tracking_software,
                input_spot_str,
                input_edge_str,
                output_path_str,
                "--fix-late-daughters",
                "--fix-missing-daughters",
                "--dead-label",
                "dead",
                "-v",
            ],
        )

        assert result.exit_code == 0
        preprocess_trackmate_or_mamut_files.assert_called_once_with(
            input_spot_path,
            input_edge_path,
            tracks_out_path,
            fix_late_daughters=True,
            fix_missing_daughters=True,
            dead_label="dead",
        )


def test_mastodon_cli(mastodon_test_data_dir, tracks_out_path):
    """Test CLI call to mastodon calls preprocess_mastodon_files with correct parameters."""
    runner = CliRunner()

    with mock.patch(
        "track_converter.src.convert_tracks.preprocess_mastodon_files", autospec=True
    ) as preprocess_mastodon_files:
        input_spot_path = mastodon_test_data_dir / "MastodonTable-Spot.csv"
        input_link_path = mastodon_test_data_dir / "MastodonTable-Link.csv"
        input_spot_str = str(input_spot_path.resolve())
        input_link_str = str(input_link_path.resolve())
        output_path_str = str(tracks_out_path.resolve())

        result = runner.invoke(
            convert_tracks,
            [
                "mastodon",
                input_spot_str,
                input_link_str,
                output_path_str,
                "--fix-late-daughters",
                "--fix-missing-daughters",
                "--dead-tagset",
                "dead_cells",
                "--dead-tag",
                "dead",
                "-v",
            ],
        )

        assert result.exit_code == 0
        preprocess_mastodon_files.assert_called_once_with(
            input_spot_path,
            input_link_path,
            tracks_out_path,
            fix_late_daughters=True,
            fix_missing_daughters=True,
            dead_tagset="dead_cells",
            dead_tag="dead",
        )


def test_ctc_cli(ctc_test_data_dir, tracks_out_path):
    """Test CLI call to CTC calls preprocess_ctc_file with correct parameters."""
    runner = CliRunner()

    with mock.patch("track_converter.src.convert_tracks.preprocess_ctc_file", autospec=True) as preprocess_ctc_file:
        input_txt_path = ctc_test_data_dir / "tracks_two_trees.txt"
        input_txt_str = str(input_txt_path.resolve())
        output_path_str = str(tracks_out_path.resolve())

        result = runner.invoke(
            convert_tracks,
            [
                "ctc",
                input_txt_str,
                output_path_str,
                "--fix-late-daughters",
                "--fix-missing-daughters",
                "--no-right-censor",
                "--dead-cell-labels",
                "[2, 5]",
                "-v",
            ],
        )

        assert result.exit_code == 0
        preprocess_ctc_file.assert_called_once_with(
            input_txt_path,
            tracks_out_path,
            fix_late_daughters=True,
            fix_missing_daughters=True,
            default_right_censor=False,
            dead_cell_labels=[2, 5],
        )
