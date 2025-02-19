
# source('./scr/all_impact_console.R')
library(tidyverse)
library(mlr3verse)
library(iml)
library(mlr3hyperband)
library(mlr3tuning)
library(mlr3learners)
library(mlr3pipelines)
library(ranger)
library(xgboost)
library(mlr3tuningspaces)

load('./dat/super_task.rda')
load('./dat/grd.rda')

 cmb_feature_groups <- list(
   ap = grep('ap_',colnames(super_task), value = T) ,
   mk = grep('mk_',colnames(super_task), value = T) ,
   pk = grep('pk_',colnames(super_task), value = T) ,
   st = grep('st_',colnames(super_task), value = T) ,
   fk = colnames(dplyr::select(super_task, fk_Flow_rate:fk_EC, -fk_O2)))

message("cmb_feature_groups done")

if(T){# if we want a small test set
  idx <- sample(nrow(grd), 10)
  grd_sample <- grd[idx, ]
  # bmr_a_sample <- bmr_a[idx, ]
  # bmr_sample <- bmr[idx]
} # if we want a small test set

message("grd_sample done", length(idx))

all_imp_mcl <- mclapply(1:nrow(grd_sample),function(i){
  task_tmp <- grd_sample[i, 'task'][[1]][[1]]$clone(deep = T)
  learner_tmp <- grd_sample[i, 'learner'][[1]][[1]]$clone(deep = T)
  if(grepl('_rg_', learner_tmp$id)){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
  feats = NULL
  if(grepl('_cmb', task_tmp$id)){feats <- cmb_feature_groups}
  task_feature_names <- task_tmp$feature_names
  iml_lst <- mlr_lst <- ale_lst <- list() # igasse kogume 10 hinnangut
  message("Processing item ", i)
  for(j in 1:10){
    message("Processing j ", j)
    learner_tmp$train(task_tmp)
    imp.pred <- iml::Predictor$new(model = learner_tmp, data = task_tmp$data(), y = task_tmp$target_names)
    
    mlr_lst[[j]] <- learner_tmp$base_learner()$importance()
    iml_lst[[j]] <- try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results %>% dplyr::select(feature, importance), silent = T)
    
    eff_lst <- list()
    for(e in 1:length(task_feature_names)){eff_lst[[e]] <- iml::FeatureEffect$new(predictor = imp.pred, feature = task_feature_names[e], grid.size = 60)$results %>% filter(.class ==1)}
    names(eff_lst) <- task_feature_names
    tmp <- lapply(eff_lst, function(x){colnames(x)[3:4] <- c('value','X'); return(dplyr::select(x, value, X))})
    eff_df <- bind_rows(tmp, .id = 'feature')
    ale_lst[[j]] <- eff_df
  } # end j loop
  return(list(bind_rows(mlr_lst, .id = 'id'), bind_rows(iml_lst, .id = 'id'), bind_rows(ale_lst, .id = 'id')))
}, mc.cores = 10)

save(all_imp_mcl, file = './dat/all_imp_mcl.rda')

