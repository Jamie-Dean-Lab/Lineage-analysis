import re
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pandas as pd
import pytest

from track_converter.src.preprocess_ctc import discard_related_cells, preprocess_ctc_file


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.resolve() / "data"


def read_tracks(tracks_path: Path):
    tracks = pd.read_table(tracks_path, sep=r"\s+", header=None)
    if len(tracks.columns) == 4:
        tracks.columns = ["L", "B", "E", "P"]
    elif len(tracks.columns) == 5:
        tracks.columns = ["L", "B", "E", "P", "R"]
    return tracks


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
def test_validate_tracks(tracks_file, expected_warning, expected_remaining_labels, tmp_path, test_data_dir, caplog):
    """
    Test validation of tracks files with various issues.

    For each, check the correct cells are removed and the correct warning is given.
    """
    tracks_in_path = test_data_dir / tracks_file
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


def test_valid_file_passes(tmp_path, test_data_dir):
    """Run a larger valid example file through the validation, checking no exceptions are raised."""
    tracks_in_path = test_data_dir / "tracks_standard_format.txt"
    tracks_out_path = tmp_path / "tracks_out.txt"

    preprocess_ctc_file(tracks_in_path, tracks_out_path)


@pytest.mark.parametrize(
    "tracks_file,labels_to_discard,expected_remaining_labels,expected_exception",
    [
        # tracks where all cells are connected in one 'tree' with structure:
        #    1
        #   / \
        #  2   3
        #     / \
        #    4   5
        pytest.param(
            "tracks_one_tree.txt",
            (5, 2),
            (),
            pytest.raises(ValueError, match=r"No tracks remaining after discarding related cells of \(5, 2\)"),
            id="one tree",
        ),
        # tracks where cells are connected into two 'trees' with structure:
        #    1       4
        #   / \     / \
        #  2   3   5   6
        pytest.param("tracks_two_trees.txt", (2,), (4, 5, 6), does_not_raise(), id="two trees"),
    ],
)
def test_discard_related_cells(
    tracks_file, labels_to_discard, expected_remaining_labels, expected_exception, test_data_dir
):
    """Test discarding all related cells in different cell lineages."""
    tracks = read_tracks(test_data_dir / tracks_file)

    with expected_exception:
        tracks = discard_related_cells(tracks, labels_to_discard)
        remaining_labels = tracks["L"]

        assert len(remaining_labels) == len(expected_remaining_labels)
        if len(expected_remaining_labels) > 0:
            assert remaining_labels.isin(expected_remaining_labels).all()
