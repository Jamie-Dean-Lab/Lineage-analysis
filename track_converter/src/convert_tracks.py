from pathlib import Path
from typing import ClassVar

import click

from track_converter.src.preprocess_btrack import preprocess_btrack_file
from track_converter.src.preprocess_trackmate_or_mamut import preprocess_trackmate_or_mamut_files


class AliasedGroup(click.Group):
    """
    Class to allow mamut as an alias to trackmate in the CLI.

    Both software share the same csv files and so have identical options / processing.
    """

    aliases: ClassVar[dict[str, str]] = {"mamut": "trackmate"}

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Find the relevant Command object if it exists."""
        if cmd_name in self.aliases:
            cmd_name = self.aliases[cmd_name]

        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        """Return a list of subcommand names in the order they should appear in --help."""
        commands = super().list_commands(ctx)
        commands.extend(self.aliases.keys())
        return commands


@click.group(cls=AliasedGroup)
def convert_tracks() -> None:
    """Overall CLI entry point - sub-commands for each file type are assigned to this group."""


@click.command()
@click.argument("h5-path")
@click.argument("output-txt-path")
@click.option(
    "--no-terminate-fates",
    is_flag=True,
    default=False,
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


@click.command()
@click.argument("spots-csv-path")
@click.argument("edges-csv-path")
@click.argument("output-txt-path")
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
    "--dead-label",
    help="Name of manually labelled 'dead' spots (will be marked as not right-censored)",
)
def trackmate(
    spots_csv_path: str,
    edges_csv_path: str,
    output_txt_path: str,
    fix_late_daughters: bool,
    fix_missing_daughters: bool,
    dead_label: str,
) -> None:
    """
    Convert Trackmate or MaMuT csv files into a text file.

    TrackMate and MaMuT share the same output csv file formats and so can be processed in the
    same way. All cells that don't end in cell division will be marked as right-censored, except for (optionally)
    those manually labelled as dead in with dead_label.
    """
    preprocess_trackmate_or_mamut_files(
        Path(spots_csv_path),
        Path(edges_csv_path),
        Path(output_txt_path),
        fix_late_daughters,
        fix_missing_daughters,
        dead_label,
    )


convert_tracks.add_command(btrack)
convert_tracks.add_command(trackmate)


if __name__ == "__main__":
    convert_tracks()
