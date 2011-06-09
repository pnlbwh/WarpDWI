#!/bin/csh
src/WarpVolume --resample_self --input /spl_unsupported/pnlfs/reckbo/projects/CreateDWIAtlas/tests/input/01019-Rgd-norm.nhdr --output resampled_dwi_at_original_directions.nrrd $*
