from pathlib import Path

import pandas as pd


def read_tracks(tracks_path: Path):
    tracks = pd.read_table(tracks_path, sep=r"\s+", header=None)
    if len(tracks.columns) == 4:
        tracks.columns = ["L", "B", "E", "P"]
    elif len(tracks.columns) == 5:
        tracks.columns = ["L", "B", "E", "P", "R"]
    return tracks
