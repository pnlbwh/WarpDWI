#!/bin/bash
DWI=dwi.nrrd
DEFORMATION=dwi_deformation.nrrd
../WarpDWI/WarpDWI --input $DWI --warp $DEFORMATION --output dwi_warped_to_canonical_directions.nrrd $*
