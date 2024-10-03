import re
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
        ("tracks_begin_larger_than_end_frame_by_one.txt", "", (1, 2, 3, 4, 5, 6)),
        (
            "tracks_begin_larger_than_end_frame_by_two.txt",
            r"All cell start frames must be <= their end frame.*invalid labels: \[2\]",
            (4, 5, 6),
        ),
    ],
)
def test_validate_track_begin_end_frames(
    tracks_file, expected_warning, expected_remaining_labels, tmp_path, test_data_dir, caplog
):
    """Test removal of invalid cells for begin frame > than end, unless it is by a single frame."""
    tracks_in_path = test_data_dir / tracks_file
    tracks_out_path = tmp_path / "tracks_out.txt"
    preprocess_ctc_file(tracks_in_path, tracks_out_path)

    # Check expected warning is given
    pattern = re.compile(expected_warning)
    assert pattern.search(caplog.text) is not None

    # Check correct cell labels are filtered out
    if tracks_out_path.exists:
        processed_tracks = read_tracks(tracks_out_path)
        remaining_labels = processed_tracks["L"]

        assert len(remaining_labels) == len(expected_remaining_labels)
        if len(expected_remaining_labels) > 0:
            assert remaining_labels.isin(expected_remaining_labels).all()


def test_validate_n_daughters(tmp_path, test_data_dir):
    """Test exception is raised when number of daughters is not equal to 2."""
    tracks_in_path = test_data_dir / "tracks_one_daughter.txt"
    tracks_out_path = tmp_path / "tracks_out.txt"

    with pytest.raises(ValueError, match=r"Cell must have 2 daughters, or None"):
        preprocess_ctc_file(tracks_in_path, tracks_out_path)


def test_validate_mother_daughter_frames(tmp_path, test_data_dir):
    """Test exception is raised when daughters don't appear one frame after the mother ends."""
    tracks_in_path = test_data_dir / "tracks_daughter_two_frames_after_mother.txt"
    tracks_out_path = tmp_path / "tracks_out.txt"

    with pytest.raises(ValueError, match=r"Daughter cells must appear one frame after"):
        preprocess_ctc_file(tracks_in_path, tracks_out_path)


def test_validate_right_censored_daughters(tmp_path, test_data_dir):
    """Test exception is raised when right-censored cells have daughters."""
    tracks_in_path = test_data_dir / "tracks_right_censored_with_daughters.txt"
    tracks_out_path = tmp_path / "tracks_out.txt"

    with pytest.raises(ValueError, match=r"Right-censored cells should have no daughters"):
        preprocess_ctc_file(tracks_in_path, tracks_out_path)


def test_valid_file_passes(tmp_path, test_data_dir):
    """Run a valid example file through the validation, checking no exceptions are raised."""
    tracks_in_path = test_data_dir / "tracks_standard_format.txt"
    tracks_out_path = tmp_path / "tracks_out.txt"

    preprocess_ctc_file(tracks_in_path, tracks_out_path)


@pytest.mark.parametrize(
    "tracks_file,labels_to_discard,expected_remaining_labels",
    [
        # tracks where all cells are connected in one 'tree' with structure:
        #    1
        #   / \
        #  2   3
        #     / \
        #    4   5
        ("tracks_one_tree.txt", (5, 2), ()),
        # tracks where cells are connected into two 'trees' with structure:
        #    1       4
        #   / \     / \
        #  2   3   5   6
        ("tracks_two_trees.txt", (2,), (4, 5, 6)),
    ],
)
def test_discard_related_cells(tracks_file, labels_to_discard, expected_remaining_labels, test_data_dir):
    """Test discarding all related cells in different cell lineages."""
    tracks = read_tracks(test_data_dir / tracks_file)
    tracks = discard_related_cells(tracks, labels_to_discard)

    remaining_labels = tracks["L"]
    assert len(remaining_labels) == len(expected_remaining_labels)
    if len(expected_remaining_labels) > 0:
        assert remaining_labels.isin(expected_remaining_labels).all()
