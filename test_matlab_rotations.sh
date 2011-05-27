#!/bin/sh
mat_rotations='/spl_unsupported/pnlfs/reckbo/projects/CreateDWIAtlas/tests/input/01019-Rgd-fa-Rotation.mat'
src/WarpVolume --without_baselines --resample --mat_rotations $mat_rotations --input /spl_unsupported/pnlfs/reckbo/projects/CreateDWIAtlas/tests/input/01019-Rgd-norm.nhdr --output resampled_dwi_nobaselines_mat_rotations.nrrd $*
