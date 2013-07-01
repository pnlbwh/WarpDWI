#!/bin/bash
DWI=dwi.nrrd
../WarpDWI/WarpDWI --resample original --input $DWI --output resampled_dwi_at_original_directions.nrrd $*
