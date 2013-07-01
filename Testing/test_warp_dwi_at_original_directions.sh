#!/bin/bash
DWI=dwi.nrrd
DEFORMATION=dwi_deformation.nrrd
../WarpDWI/WarpDWI --resample original --input $DWI --warp $DEFORMATION --output dwi_warped_to_original_directions.nrrd $*
