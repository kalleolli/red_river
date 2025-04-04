#!/bin/sh
#
# Run R script with Linux script folder as working directory
#
# Determine the folder where this Linux shell script lives
# rwd=$(cd $(dirname "$0") && pwd)
# Run the R script
# - R is probably in the PATH since it installs to a common bin folder
# - the first use of $rwd tells R where to find the R script
# - the second use of $rwd tells the script being run where the script exists
# - all paths in the R script should be absolute using %wd%
# - a more explicit command line option than the first R script argument could be implemented
# Rscript "${scriptFolder}/test_1.R" "${rwd}" "${rwd}"


 Rscript scr/red_river_1.R
 Rscript scr/red_river_2.R
 Rscript scr/red_river_3.R
 Rscript scr/red_river_4.R
Rscript scr/red_river_5.R # memory hungry