All mastodon test files were made using one of the example images that comes with Fiji: `File > Open Samples > Tracks for TrackMate` with the frame interval set to `5 sec` and the pixel width / height / depth set to `5 microns` (under `Image > Properties`). This was then saved as a bdv format file using `Plugins > BigDataViewer > Export Current Image as XML/HDF5`

I then ran through the standard Mastodon workflow using `Plugins > Tracking > Detection` with the `LoG detector` (estimated object diameter of 25 microns and quality threshold of 10.0). Then, `Plugins > Tracking > Linking` with the `LAP linker` (allowing gap closing and track division, but not track fusion. All distances set to 50 microns).

For two cells, the name of their last spot was changed to 'dead' by opening `TrackScheme`, right clicking on the spot and selecting `Edit spot name`
