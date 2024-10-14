from pathlib import Path

import click

from track_converter.src.preprocess_btrack import preprocess_btrack_file


@click.group()
def convert_tracks() -> None:
    """Overall CLI entry point - sub-commands for each file type are assigned to this group."""


@click.command()
@click.argument("h5-path")
@click.argument("output-txt-path")
@click.option(
    "--no-terminate-fates",
    is_flag=True,
    default=True,
    help="Don't use btrack's 'TERMINATE' fates to determine right-censoring",
)
@click.option(
    "--keep-false-positives",
    is_flag=True,
    default=False,
    help="Keep cells with a btrack fate of FALSE_POSITIVE",
)
@click.option(
    "--fix-late-daughters",
    is_flag=True,
    default=False,
    help="Back-date any late daughters to the start time of the earlier daughter.",
)
@click.option(
    "--fix-missing-daughters",
    is_flag=True,
    default=False,
    help="Create a second daughter for any mother cells that only have one.",
)
@click.option(
    "--dead-track-ids",
    multiple=True,
    type=int,
    help="List of track ids (.ID for each btrack Tracklet) to consider as 'dead' cells (i.e. not right-censored)",
)
def btrack(
    h5_path: str,
    output_txt_path: str,
    no_terminate_fates: bool,
    keep_false_positives: bool,
    fix_late_daughters: bool,
    fix_missing_daughters: bool,
    dead_track_ids: tuple[int],
) -> None:
    """
    Convert a btrack output file (.h5) into a text file.

    By default, all right censoring information will be read directly from btrack's assigned 'fates' and cells with
    'false positive' fates will be removed. To disable this, use the --no-terminate-fates and --keep-false-positives
    options. With --no-terminate-fates, all cells that don't end in cell division will be marked as right-censored
    except for (optionally) those provided with --dead-track-ids.
    """
    preprocess_btrack_file(
        Path(h5_path),
        Path(output_txt_path),
        not no_terminate_fates,
        not keep_false_positives,
        fix_late_daughters,
        fix_missing_daughters,
        dead_track_ids,
    )


convert_tracks.add_command(btrack)


if __name__ == "__main__":
    convert_tracks()
