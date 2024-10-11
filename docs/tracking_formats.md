# Handling different cell tracking formats

This tool is compatible with the output of the following software:
- [TrackMate](https://imagej.net/plugins/trackmate/)
- [MaMuT](https://imagej.net/plugins/mamut/)
- [Mastodon](https://imagej.net/plugins/mastodon)
- [btrack](https://btrack.readthedocs.io/en/latest/)

or any files in the [cell tracking challenge format](https://celltrackingchallenge.net/datasets/).

Instructions for exporting the required files from each are provided below:

## TrackMate

### Right-censoring

By default, the tool will assume that all cells that don't end in cell division are 'right-censored' (i.e. their fate 
is unknown because they went out of frame, reached the end of the video etc.). If you want to mark specific cells as 
dead (i.e. not right-censored), you can do this via TrackMate:

- Click on `TrackScheme` to view all tracks
  
- Find the final 'spot' for a dead cell track - either by browsing in `TrackScheme` or in the main image window. 
  Note that clicking on a spot in the image window will snap `Trackscheme` to that location.
  
- Right click on the spot in `TrackScheme` and select `Edit spot name` (note that you can view all spot names by 
  changing the `Style` in the top menu bar of `TrackScheme` to `full`).
  
- Use the same name for all dead cells e.g. `dead`

### Exporting required csv files

The tool requires two csv files from TrackMate: `spots.csv` and `edges.csv`. These can be exported by clicking the 
`Tracks` button, then the `Spots` tab on the left hand side, followed by `Export to csv`. 
Repeat this for the `Edges` tab.


## MaMuT

MaMuT files are handled in the same way as TrackMate. See the [`TrackMate` section](#trackmate) for information on how 
to export the required files (note that in MaMuT the `Tracks` button is renamed to `Track tables`).

## Mastodon

### Right-censoring

As with TrackMate, by default the tool will assume that all cells that don't end in cell division are 'right-censored'.
If you want to mark specific cells as dead (i.e. not right-censored), you can do this via [Mastodon's tags
](https://mastodon.readthedocs.io/en/latest/docs/partA/numerical_features_tags_the_table_view.html#tags-and-tag-sets):

- Click `configure tags` in the main Mastodon menu

- Click the green plus icon on the left to add a 'tag set' with a name like `dead_cells`
  
- Click the green plus icon on the right to add a 'tag' to this 'tagset' e.g. `dead`

- In any mastodon view (bdv / TrackScheme / tables), select a spot or multiple spots then select 
  `Edit > Tags > tagset_name > tag_name` from the top menubar. The spots you select should be the final spot of a dead 
  cell track. Note: in TrackScheme you can also click `Y`, then use the number keys to select the right tag.

- To colour the view by your tags, you can select `View > Coloring > tagset_name` in the top menubar.

### Exporting required csv files

The tool requires two csv files from Mastodon: `Spot.csv` and `Link.csv`. These can be exported by clicking the 
`table` button, then the `Spot` tab on the left hand side, followed by `File > Export to CSV` in the top menubar. 
Repeat this for the `Link` tab.

## btrack

### Right-censoring

By default, the tool will use btrack's assigned 'fates' to determine the right-censoring status of all cells. If you 
want to set this manually instead, you can provide a list of track ids (accessed via `.ID` for each btrack Tracklet) 
to mark as dead cells (i.e. not right-censored).

### Exporting required csv files

The tool expects the standard `.h5` output from btrack e.g. from 
`tracker.export('/path/to/tracks.h5', obj_type='obj_type_1')`


## Cell tracking challenge

If your files are already in the cell tracking challenge format, then no further conversion is necessary. The tool only 
requires the txt file representing an acyclic graph for the whole video (e.g. `man_track.txt` / `res_track.txt`).

