from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest

from track_converter.src.preprocess_ctc import preprocess_ctc_file


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.resolve() / "data"


@pytest.mark.parametrize(
    "tracks_file,expectation",
    [
        ("tracks_begin_larger_than_end_frame_by_one.txt", does_not_raise()),
        (
            "tracks_begin_larger_than_end_frame_by_two.txt",
            pytest.raises(ValueError, match=r"tracks start frame must be <= the tracks end frame"),
        ),
    ],
)
def test_validate_track_begin_end_frames(tracks_file, expectation, tmp_path, test_data_dir):
    """Test exception is raised for begin frame > than end, unless it is by a single frame."""
    tracks_in_path = test_data_dir / tracks_file
    tracks_out_path = tmp_path / "tracks_out.txt"

    with expectation:
        preprocess_ctc_file(tracks_in_path, tracks_out_path)


def test_validate_n_daughters(tmp_path, test_data_dir):
    """Test exception is raised when number of daughters is not equal to 2."""
    tracks_in_path = test_data_dir / "tracks_one_daughter.txt"
    tracks_out_path = tmp_path / "tracks_out.txt"

    with pytest.raises(ValueError, match=r"Tracks must have 2 daughters, or None"):
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
