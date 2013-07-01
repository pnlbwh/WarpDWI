#!/bin/bash
DWI=dwi.nrrd
../WarpDWI/WarpDWI --without_baselines --resample canonical --input $DWI --output resampled_dwi_at_canonical_directions_nobaselines.nrrd $*
