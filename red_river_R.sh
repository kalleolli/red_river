#!/bin/sh
#
# Run R scripts at ../scr folder as working directory
#

 nohup nice -n 19  R CMD BATCH --no-restore scr/red_river_1.R ./log/red_river_1.txt
 nohup nice -n 19  R CMD BATCH --no-restore scr/red_river_2.R ./log/red_river_2.txt
 nohup nice -n 19  R CMD BATCH --no-restore scr/red_river_3.R ./log/red_river_3.txt
 nohup nice -n 19  R CMD BATCH --no-restore scr/red_river_4.R ./log/red_river_4.txt
 nohup nice -n 19  R CMD BATCH --no-restore scr/red_river_5.R ./log/red_river_5.txt
 nohup nice -n 19  R CMD BATCH --no-restore scr/red_river_6_figs1.R ./log/red_river_6.txt
 nohup nice -n 19  R CMD BATCH --no-restore scr/red_river_7_figs2.R ./log/red_river_7.txt
 nohup nice -n 19  R CMD BATCH --no-restore scr/red_river_8_figs3.R ./log/red_river_8.txt
# Alternatives:
# Rscript scr/red_river_1.R
# Rscript scr/red_river_2.R
# Rscript scr/red_river_3.R
# Rscript scr/red_river_4.R
# Rscript scr/red_river_5.R # memory hungry
# Rscript scr/red_river_6_figs1.R 
# Rscript scr/red_river_7_figs2.R 
# Rscript scr/red_river_8_figs3.R 

