#!/bin/csh
src/WarpVolume --without_baselines --resample --input /spl_unsupported/pnlfs/reckbo/projects/CreateDWIAtlas/tests/input/01019-Rgd-norm.nhdr --output resampled_dwi_nobaselines.nrrd $*
