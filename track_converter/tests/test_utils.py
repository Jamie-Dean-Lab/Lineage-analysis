from contextlib import nullcontext as does_not_raise

import pytest

from track_converter.src.utils import discard_related_cells
from track_converter.tests.utils import read_tracks


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
    tracks_file, labels_to_discard, expected_remaining_labels, expected_exception, ctc_test_data_dir
):
    """Test discarding all related cells in different cell lineages."""
    tracks = read_tracks(ctc_test_data_dir / tracks_file)

    with expected_exception:
        tracks = discard_related_cells(tracks, labels_to_discard)
        remaining_labels = tracks["L"]

        assert len(remaining_labels) == len(expected_remaining_labels)
        if len(expected_remaining_labels) > 0:
            assert remaining_labels.isin(expected_remaining_labels).all()
