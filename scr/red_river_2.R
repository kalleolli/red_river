
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
load(paste0(reddir, 'dat/red_river_1.rda'))

for(i in 1:length(tsks)){ # 6 CPUs but each takes GB's of memory and swap
  hyptune <-  parallel::mclapply(glrn_xx_lst, function(x){
    try(mlr3tuning::tune(
      tuner = mlr3tuning::tnr("hyperband", eta = 2, repetitions = 1), 
      task = tsks[[i]], 
      learner = x, # saame siia kõik 6 learnerit suruda
      resampling = mlr3::rsmp("cv", folds = 3),
      measures = mlr3::msr("classif.ce"),
      terminator = bbotk::trm("none")), silent = T )}, mc.cores = 6) # siit on vaja salvestada ainult headerid
  
  hyptune_heads <- lapply(lapply(hyptune, '[[', 'archive'), '[[', 'data') %>% lapply(., function(x){arrange(x, classif.ce) %>% head(50) %>% dplyr::select(classif.ce, x_domain)})
  
  savedir <- paste0(reddir, 'dat/hyptune_', names(tsks)[i],'.rda')
  message("Task ", names(tsks)[i] ," hyperband tuning completed \n")
  save(hyptune_heads, file = savedir)}

message("All hyperband tunign completed \n")

# provides 12 hyptune2_tsk_*.rda header files in ./dat folder