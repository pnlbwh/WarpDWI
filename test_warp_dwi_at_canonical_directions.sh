#!/bin/csh
#src/WarpVolume --input /spl_unsupported/pnlfs/reckbo/projects/CreateDWIAtlas/tests/input/01019-Rgd-norm.nhdr --warp /spl_unsupported/pnlfs/reckbo/data/10_masked_cases_affined/G119_128_0_deformation.nii.gz --output warped_DWI_volume.nrrd $*
src/WarpVolume --input 01019.nhdr --warp deformation.nrrd --output warped_DWI_at_canonical_directions.nrrd $*
