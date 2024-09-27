from pathlib import Path

import pandas as pd


def validate_tracks_shape_dtypes(tracks: pd.DataFrame) -> None:
    """
    Validate the shape (number of rows / columns) and data types / ranges of columns.

    This will also add a final right-censoring flag column (defaulting to all zero) if it doesn't exist.
    """
    nrows = tracks.shape[0]
    ncols = tracks.shape[1]

    if ncols not in (4, 5):
        msg = "input file must have 4 or 5 columns"
        raise ValueError(msg)

    if nrows == 0:
        msg = "input file must contain at least one track"
        raise ValueError(msg)

    # add right-censoring column (if it doesn't exist), defaulting to all zero
    if ncols == 4:
        tracks["R"] = 0

    # Re-name columns to LBEPR for easy access
    tracks.columns = ["L", "B", "E", "P", "R"]

    # Check data type and ranges of columns
    cols_are_integers = tracks.apply(pd.api.types.is_integer_dtype, axis=0)
    if not cols_are_integers.all():
        msg = "all columns must contain integer values"
        raise ValueError(msg)

    if not (tracks["L"] > 0).all():
        msg = "all values in the first column must be greater than zero"
        raise ValueError(msg)

    if not (tracks["P"] >= 0).all():
        msg = "all values in the fourth column must be greater than or equal to zero"
        raise ValueError(msg)

    if not tracks["R"].isin((0, 1)).all():
        msg = "all values in the last column must be 0 or 1"
        raise ValueError(msg)


def preprocess_ctc_file(input_ctc_filepath: Path, output_ctc_filepath: Path) -> None:
    """
    Preprocess Cell Tracking Challenge (CTC) format files.

    Expects files in the standard CTC format, optionally with an additional column indicating right censoring.
    Columns are L B E P (R):
    L - a unique label of the track (any positive number, not zero or negative)
    B - the frame where the track begins (integer)
    E - the frame where the track ends (integer)
    P - label of the parent track (0 is used when the parent is unknown)
    R - right censoring flag (1=right-censored, 0=not). Note a 0 only means it is not manually declared as
    right-censored. It will still be considered right-censored by the processing code, if the last observed frame
    coincides with the end of the movie.
    """
    tracks = pd.read_table(input_ctc_filepath, sep=r"\s+", header=None)
    validate_tracks_shape_dtypes(tracks)

    # save new file
    tracks.to_csv(output_ctc_filepath, sep=" ", header=False, index=False)
