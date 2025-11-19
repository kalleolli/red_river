
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

# hyperband tuning of learners on 12 tasks


if(!exists('reddir')){
reddir <- list.files(path = '~/Documents', full.names = TRUE, recursive = TRUE, pattern = 'red_river.Rproj') %>% dirname()
}

load(paste0(reddir, '/dat/red_river_1.rda')) # load tsks, glrn_xx_lst, which were created in red_river_1.R

for(i in 1:length(tsks)){ 
  hyptune <-  parallel::mclapply(glrn_xx_lst, function(x){ # x is graph learner; we have 6 of them. 6 CPUs - each takes GB's of memory and swap
    try(mlr3tuning::tune(
      tuner = mlr3tuning::tnr("hyperband", eta = 2, repetitions = 1), 
      task = tsks[[i]], 
      learner = x,
      resampling = mlr3::rsmp("cv", folds = 3),
      measures = mlr3::msr("classif.ce"),
      terminator = bbotk::trm("none")), silent = T )}, mc.cores = 6) 
  
  # from here we only save top 50 hyperparameter sets per learner
  hyptune_heads <- lapply(lapply(hyptune, '[[', 'archive'), '[[', 'data') %>% lapply(., function(x){arrange(x, classif.ce) %>% head(50) %>% dplyr::select(classif.ce, x_domain)})
  
  savedir <- paste0(reddir, '/dat/hyptune_', names(tsks)[i],'.rda')
  message("Task ", names(tsks)[i] ," hyperband tuning completed \n")
  save(hyptune_heads, file = savedir)}

message("All hyperband tunign completed \n")
message(" proceed with red_river_3.R \n")

# provides 12 hyptune2_tsk_*.rda header files in ./dat folder
