
library(mlr3)
library(mlr3hyperband)
library(mlr3tuning)
library(mlr3learners)
library(mlr3pipelines)
library(iml)
library(ranger)
library(xgboost)

library(mlr3tuningspaces)
library(mlr3extralearners)
library(dplyr)
library(tidyverse)

reddir <- system("find ~/Documents -name 'red_river.Rproj' ", intern = T) %>% sub('red_river.Rproj','', .)
load(paste0(reddir, 'dat/grd2.rda'))
 
message("Actual benchmarking with future speedup \n\n ")
  
library(future)
plan(multisession) # speeds up benchmarking

bmr <- mlr3::benchmark(grd)

bmr_a <- bmr$aggregate();
# parse bmr categories; delete capital D
bmr_a <- strsplit(bmr_a$task_id, '_') %>% do.call(rbind, .) %>% as.data.table() %>% transmute(., red_id = sub('D','',V1), feat_id = V2) %>% bind_cols(bmr_a, .);
# table(bmr_a$feat_id) # all 604
bmr_a <- strsplit(bmr_a$learner_id, '_') %>% do.call(rbind, .) %>% as.data.table() %>% transmute(., hyp_id = V2, mod_id = sub('b','',V3), batch_id = V4) %>% bind_cols(bmr_a, .);

save(bmr, bmr_a, file = paste0(reddir, 'dat/bmr2.rda'))

