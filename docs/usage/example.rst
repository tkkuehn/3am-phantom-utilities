Usage Example
=============

To demonstrate the capabilities of dMRI Phantom Utilities, we'll perform a DTI fit of some phantom data, and use a description of the phantoms to produce ground truth images describing those phantoms.

We'll start by loading the example phantom data (available for download `here <https://openneuro.org/datasets/ds002350/>`_)::

    from dmriphantomutils import image_io
    source_dir = './resources/manual_complex_bids/sub-01/ses-01/dwi/'
    unmasked_dwi = image_io.load_dwi(
        os.path.join(source_dir, 'sub-01_ses-01_dwi.nii.gz'),
        os.path.join(source_dir, 'sub-01_ses-01_dwi.bval'),
        os.path.join(source_dir, 'sub-01_ses-01_dwi.bvec'))

To generate ground truth images of our phantoms, we'll need to use ``scan_info`` to describe their infill (and some supplementary information about the study)::

    from dmriphantomutils import scan_info
    bending = scan_info.Phantom(
        225, 30, 0.1, 100, scan_info.ConcentricArcPattern((0, 10)))
    fanning = scan_info.Phantom(
        225, 30, 0.1, 100, scan_info.AlternatingPattern(
            scan_info.ConcentricArcPattern((-5, 0)),
            scan_info.ConcentricArcPattern((5, 0))))
    kissing = scan_info.Phantom(
        225, 30, 0.1, 100, scan_info.AlternatingPattern(
            scan_info.ConcentricArcPattern((0, 6.5)),
            scan_info.ParallelLinePattern(90)))
    crossing = scan_info.Phantom(
        225, 30, 0.1, 100, scan_info.AlternatingPattern(
            scan_info.ParallelLinePattern(0),
            scan_info.ParallelLinePattern(90)))
    straight = scan_info.Phantom(
        225, 30, 0.1, 100, scan_info.ParallelLinePattern(0))
    water = scan_info.WaterSlice()

    tube = [water, kissing, bending, bending, crossing, straight]
    scan = scan_info.SingleScan(slice(0, 6))
    session = scan_info.ScanSession(datetime.date(2019, 7, 9), [scan])
    study = scan_info.Study('bending', tube, [session])

To efficiently perform our DTI fit and automatically register our ground truths, we'll want a mask for each of our phantoms. We can use ``automask`` to generate these masks::

    from dmriphantomutils import automask

    slice_masks = [automask.mask_phantom(
        unmasked_dwi.get_image()[..., slice_idx, 0]) for slice_idx in range(6)]
    phantom_masks = []
    for slice_idx in range(6):
        phantom_mask = np.zeros(unmasked_dwi.get_image().shape[0:3])
        phantom_mask[..., slice_idx] = slice_masks[slice_idx]
        phantom_masks.append(phantom_mask)
    build_dir = './build/'
    os.makedirs(build_dir, exist_ok=True)
    for mask, idx in zip(phantom_masks, range(6)):
        image_io.save_image(mask, unmasked_dwi.img.affine, unmasked_dwi.img.header,
            os.path.join(build_dir, 'mask_slice_' + str(idx) + '.nii.gz'))

    # apply masks to raw nifti
    phantom_dwis = [
        image_io.MaskedDiffusionWeightedImage(
            unmasked_dwi.img, unmasked_dwi.gtab, mask)
        for mask in phantom_masks]

Note that while ``automask`` generally does a good job filtering out any air bubbles in phantoms, it's a good idea to take a look at the b0 images and manually adjust the masks as necessary.

We'll now perform our DTI fit, and save a copy of the mean diffusivity maps::

    from dmriphantomutils import dipy_fit
    dtifits = [dipy_fit.fit_dti(dwi) for dwi in phantom_dwis]
    for fit, dwi, idx in zip(dtifits, phantom_dwis, range(6)):
        dipy_fit.save_dti_metric_imgs(
            dwi,
            fit, 
            md_path=os.path.join(build_dir, 'md_slice_' + str(idx) + '.nii.gz'))

With that done, we can use our MD map to find the fiducials in the phantom (the fiducials are holes in the phantom containing free water, so MD should be highest in the fiducial), and our mask to find each phantom's centroid::

    from dmriphantomutils import transform_data
    fiducials = [np.unravel_index(np.argmax(fit.md), fit.md.shape)[0:2]
        for fit in dtifits]
    centroids = [transform_data.find_centroid(mask[..., slice_idx])
        for mask, slice_idx in zip(phantom_masks, range(6))]

Finally, we can give ``transform_data`` the fiducials, centroids, and phantom information to produce ground truth images in image space::

    for mask, phantom, centroid, fiducial, dwi, slice_idx in zip(
            phantom_masks, tube, centroids, fiducials, phantom_dwis, range(6)):
        for metric, generator in (
                phantom.infill_pattern.get_geometry_generators().items()):
            metric_img = transform_data.gen_geometry_data(
                mask,
                generator,
                centroid,
                dwi.img.header['pixdim'][1],
                fiducial=(fiducial[0], fiducial[1]))
            image_io.save_image(
                metric_img, dwi.img.affine, dwi.img.header,
                os.path.join(
                    build_dir,
                    'slice_' + str(slice_idx) + 'metric_' + metric + '.nii.gz'))

We're left with masks, MD maps, and ground truth maps for each phantom in our source image. Using these masks, we could investigate the relationship between DTI metrics and crossing angle, for one example.

