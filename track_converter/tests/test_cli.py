from unittest import mock

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
