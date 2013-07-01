#!/bin/bash
#
# The "$*" at the end of the commandline call below are the parameters passed into
# this shell script.  For example, you can run ./test_warp_3d.sh --echo to print
# out the input parameters.
IMAGE=t1.nrrd
DEFORMATION=t1_deformation.nii.gz
../WarpDWI/WarpDWI --input $IMAGE --warp $DEFORMATION --output t1_warped.nrrd $*
