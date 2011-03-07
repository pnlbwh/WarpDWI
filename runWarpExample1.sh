#!/bin/csh
#src/WarpVolume --inputVolume /projects/schiz/3Tdata/case01009/diff/01009-dwi-filt-Ed.nhdr --warp /spl_unsupported/pnlfs/reckbo/data/10_masked_cases_affined/G119_128_0_deformation.nii.gz --resultsDirectory ./
src/WarpVolume --inputVolume /spl_unsupported/pnlfs/reckbo/projects/CreateDWIAtlas/tests/input/01019-Rgd-norm.nhdr --warp /spl_unsupported/pnlfs/reckbo/data/10_masked_cases_affined/G119_128_0_deformation.nii.gz --resultsDirectory ./
