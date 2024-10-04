import re

import numpy as np
import pandas as pd
import pytest

from track_converter.src.preprocess_ctc import preprocess_ctc_file
from track_converter.tests.utils import read_tracks


@pytest.mark.parametrize(
    "tracks_file,expected_warning,expected_remaining_labels",
    [
        # validate_cell_begin_end_frames
        pytest.param(
            "tracks_begin_larger_than_end_frame_by_one.txt",
            "",
            (1, 2, 3, 4, 5, 6),
            id="Begin frame larger than end frame by one",
        ),
        pytest.param(
            "tracks_begin_larger_than_end_frame_by_two.txt",
            r"All cell start frames must be <= their end frame.*invalid labels: \[2\]",
            (4, 5, 6),
            id="Begin frame larger than end frame by two",
        ),
        # validate_n_daughters
        pytest.param(
            "tracks_one_daughter.txt",
            r"Cells must have 2 daughters, or None. .*invalid labels: \[1\]",
            (3, 4, 5),
            id="Cell only has one daughter",
        ),
        # validate_mother_daughter_frames
        pytest.param(
            "tracks_daughter_two_frames_after_mother.txt",
            r"Daughter cells must appear one frame after .*invalid labels: \[2\]",
            (4, 5, 6),
            id="Daughter appears two frames after mother ends",
        ),
        # validate_right_censored_daughters
        pytest.param(
            "tracks_right_censored_with_daughters.txt",
            r"Right-censored cells should have no daughters..*invalid labels: \[2 3\]",
            (4, 5, 6),
            id="Right-censored cell has daughters",
        ),
    ],
)
def test_validate_tracks(tracks_file, expected_warning, expected_remaining_labels, tmp_path, ctc_test_data_dir, caplog):
    """
    Test validation of tracks files with various issues.

    For each, check the correct cells are removed and the correct warning is given.
    """
    tracks_in_path = ctc_test_data_dir / tracks_file
    tracks_out_path = tmp_path / "tracks_out.txt"
    preprocess_ctc_file(tracks_in_path, tracks_out_path)

    # Check expected warning is given
    pattern = re.compile(expected_warning)
    assert pattern.search(caplog.text) is not None, "Expected warning doesn't match"

    # Check correct cell labels are filtered out
    if tracks_out_path.exists:
        processed_tracks = read_tracks(tracks_out_path)
        remaining_labels = processed_tracks["L"]

        assert len(remaining_labels) == len(expected_remaining_labels)
        if len(expected_remaining_labels) > 0:
            assert remaining_labels.isin(expected_remaining_labels).all()


def test_valid_file_passes(tmp_path, ctc_test_data_dir):
    """Run a larger valid example file through the validation, checking no exceptions are raised."""
    tracks_in_path = ctc_test_data_dir / "tracks_valid_large.txt"
    tracks_out_path = tmp_path / "tracks_out.txt"

    preprocess_ctc_file(tracks_in_path, tracks_out_path)


@pytest.mark.parametrize(
    "tracks_file,expected_warning,expected_output",
    [
        pytest.param(
            "tracks_late_daughter.txt",
            r"Daughters with parents \[1\] don't have the same begin frame",
            # Output dataframe is identical to the input, except for cell 3 has had its begin frame reduced to 3
            pd.DataFrame(
                np.array(
                    [
                        [1, 0, 2, 0, 0],
                        [2, 3, 4, 1, 0],
                        [3, 3, 4, 1, 0],
                        [4, 2, 4, 0, 0],
                        [5, 5, 6, 4, 0],
                        [6, 5, 6, 4, 0],
                    ]
                )
            ),
            id="Daughters appear one frame apart",
        ),
        pytest.param(
            "tracks_two_trees.txt",
            "",
            # output is identical to input - just want to check that no changes are made when the tracks are valid
            pd.DataFrame(
                np.array(
                    [
                        [1, 0, 2, 0, 0],
                        [2, 3, 4, 1, 0],
                        [3, 3, 4, 1, 0],
                        [4, 0, 2, 0, 0],
                        [5, 3, 4, 4, 0],
                        [6, 3, 4, 4, 0],
                    ]
                )
            ),
            id="Daughters appear on same frame",
        ),
    ],
)
def test_fix_late_daughters(tracks_file, expected_warning, expected_output, tmp_path, ctc_test_data_dir, caplog):
    tracks_in_path = ctc_test_data_dir / tracks_file
    tracks_out_path = tmp_path / "tracks_out.txt"
    preprocess_ctc_file(tracks_in_path, tracks_out_path, fix_late_daughters=True)

    # Check expected warning is given
    pattern = re.compile(expected_warning)
    assert pattern.search(caplog.text) is not None, "Expected warning doesn't match"

    # Check output is correct
    if tracks_out_path.exists:
        processed_tracks = read_tracks(tracks_out_path)
        expected_output.columns = ["L", "B", "E", "P", "R"]
        assert processed_tracks.equals(expected_output)


@pytest.mark.parametrize(
    "tracks_file,expected_warning,expected_output",
    [
        pytest.param(
            "tracks_one_daughter.txt",
            r"Cells with label \[1\] only have one daughter",
            # Output dataframe is identical to the input, with one extra row for the generated daughter
            pd.DataFrame(
                np.array(
                    [
                        [1, 0, 2, 0, 0],
                        [2, 3, 4, 1, 0],
                        [3, 0, 2, 0, 0],
                        [4, 3, 4, 3, 0],
                        [5, 3, 4, 3, 0],
                        [6, 3, 2, 1, 1],
                    ]
                )
            ),
            id="Cell only has one daughter",
        ),
        pytest.param(
            "tracks_two_trees.txt",
            "",
            # output is identical to input - just want to check that no changes are made when the tracks are valid
            pd.DataFrame(
                np.array(
                    [
                        [1, 0, 2, 0, 0],
                        [2, 3, 4, 1, 0],
                        [3, 3, 4, 1, 0],
                        [4, 0, 2, 0, 0],
                        [5, 3, 4, 4, 0],
                        [6, 3, 4, 4, 0],
                    ]
                )
            ),
            id="All cells have two daughters",
        ),
    ],
)
def test_fix_missing_daughters(tracks_file, expected_warning, expected_output, tmp_path, ctc_test_data_dir, caplog):
    tracks_in_path = ctc_test_data_dir / tracks_file
    tracks_out_path = tmp_path / "tracks_out.txt"
    preprocess_ctc_file(tracks_in_path, tracks_out_path, fix_missing_daughters=True)

    # Check expected warning is given
    pattern = re.compile(expected_warning)
    assert pattern.search(caplog.text) is not None, "Expected warning doesn't match"

    # Check output is correct
    if tracks_out_path.exists:
        processed_tracks = read_tracks(tracks_out_path)
        expected_output.columns = ["L", "B", "E", "P", "R"]
        assert processed_tracks.equals(expected_output)
