Image preparation
=================

Having prepared a sample with a set of 3D printed phantoms for analysis, there are a few things to consider when designing a scan of those phantoms.

This package has been designed for images with anisotropic voxels, where each z-slice covers the thickness of an entire phantom. This setup using thick voxels provides a good SNR and ensures each voxel samples multiple layers of material in the phantom. The practical effect is that functions in this package that conceptually work with a single phantom expect the DWI data from that phantom to be confined to a single z-slice. It is possible to work around this expectation, dMRI Phantom Utilities will be easiest to use if the source DWIs contain one phantom per z-slice.

