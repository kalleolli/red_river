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
# benchmark ####

# hyperband tuned hyperparameter collections of 50 with the respective tasks has benchmarks

# make lightweight featureless, xg and rg learners
if(T){
  # ranger does not accept missing data, we impute median values with graph learner
  glrn_fl <- as_learner(po("imputemedian") %>>% lrn("classif.featureless", method = 'weighted.sample', predict_type = 'prob'))
  glrn_xg <- as_learner(po("imputemedian") %>>% lrn("classif.xgboost", predict_type = 'prob'))
  glrn_rg <- as_learner(po("imputemedian") %>>% lrn("classif.ranger", predict_type = 'prob', importance = 'impurity'))
  
} # provides glrn_* three light learners, rg, xg, and fl (featureless)

message("Start making a list of learners with hyperband tuned parameters included  \n\n ")

if(T){
  # make a list of learners with hyperband tuned parameters included  
  # list of 12 - each is combo of Red/Bat, x 6 feature groups (cmb, fk, ap, pk, mk, st)
  # each of 12 is 50 header x 6 learners = 300 learners per list slot
  # loop over 12 hyp_files; load (as hyptune_heads)
  
  # list of 12 files, which store the tuned hyperparameter sets
  hyp_files <- list.files(paste0(reddir,'dat/')) %>% grep('hyptune2', .,  value = T) # 12
  
  hyp_lst <- list() # siia kogume 12 hyptune learnerite setti
  for(hf in 1:length(hyp_files)){
    load(paste0(reddir,'dat/', hyp_files[hf])) # loads hyptune_heads; length 6
    # tahaks saada 6st listi, iga 50 learneriga, mille hüperparameetrid on hyptune_heads x_domain
    tmp <- lapply(seq_along(hyptune_heads), function(i){
      nimi <- names(hyptune_heads)[i]
      if(grepl('_rg', nimi)) learner1 <- glrn_rg$clone(deep = T) else learner1 <- glrn_xg$clone(deep = T)
      lst <- lapply(1:nrow(hyptune_heads[[i]]), function(x){learner2 <- learner1$clone(deep = T); learner2$param_set$set_values(.values = hyptune_heads[[i]]$x_domain[[x]]);  learner2$id <- paste(nimi, x, sep = '_'); return(learner2)})
      return(lst)
    })
    hyp_lst[[hf]] <- unlist(tmp)
    message('Processing hypfile ', hf, '\n' )
  }
  names(hyp_lst) <- sub('hyptune2_tsk_','', hyp_files) %>% sub('.rda','',.)
  
} # provides hyp_lst - hyperparameter loaded learners, ready to be benchmarked


message("Start making a list of of benchmark grids  \n ")

if(T){
  # names(tsks) # from tasks_on_the_fly
  
  # benchmark grid - hyp_lst
  glrn_xg$id <- 'glrn_0_xg_0'
  glrn_rg$id <- 'glrn_0_rg_0'
  
  ## list of benchmark grids
  grd_lst <- list()
  for(i in 1:length(hyp_lst)){
    grd_lst[[i]] <- mlr3::benchmark_grid(
      tasks = tsks[i],
      learners = c(glrn_rg, glrn_xg, hyp_lst[[i]]),
      rsmp('cv', folds = 5))
    message(' Processing grd_lst ', i, '\n')
  }
  # unlist
  grd <- bind_rows(grd_lst) # 3624  x  3 benchmark grid
  save(grd, file = './dat/grd2.rda') 
  
} # provides and saves 3624  x  3 ./dat/grd2.rda - benchmark grid object

message("Benchmark grid saved in ./dat/grd2.rda  \n\n ")


