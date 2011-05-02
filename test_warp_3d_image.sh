#!/bin/csh
#
# The "$*" at the end of the commandline call below are the parameters passed into
# this shell script.  For example, you can run ./test_warp_3d.sh --echo to print
# out the input parameters.
src/WarpVolume --input /spl_unsupported/pnlfs/reckbo/data/10_masked_cases_affined/G119_128.nrrd --warp /spl_unsupported/pnlfs/reckbo/data/10_masked_cases_affined/G119_128_0_deformation.nii.gz --output warped_3D_image.nrrd $*
