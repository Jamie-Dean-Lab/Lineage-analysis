All trackmate test files were made using one of the example images that comes with Fiji: `File > Open Samples > Tracks for TrackMate` with the frame interval set to `5 sec` and the pixel width / height / depth set to `5 microns` (under `Image > Properties`).

I then ran through the standard TrackMate workflow using the `LoG detector` (with an estimated object diameter of 25 microns) followed by the `LAP Tracker` (allowing gap closing and track segment splitting, but not track segment merging. All distances set to 50 microns).

For two cells, the name of their last spot was changed to 'dead' by opening `TrackScheme`, right clicking on the spot and selecting `Edit spot name`
