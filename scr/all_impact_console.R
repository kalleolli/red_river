
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
library(parallel)

load('./dat/super_task.rda')
load('./dat/grd2.rda')

  
names(super_task) %>% grep('fk_',., value = T)
# võtame super_taskist välja "fk_Flow_rate"   "fk_NH4"  "fk_O2" "fk_COD" 
super_task <- dplyr::select(super_task, -c("fk_Flow_rate","fk_NH4","fk_O2","fk_COD","fk_NO3"))
 cmb_feature_groups <- list(
   ap = grep('ap_',colnames(super_task), value = T) ,
   mk = grep('mk_',colnames(super_task), value = T) ,
   pk = grep('pk_',colnames(super_task), value = T) ,
   st = grep('st_',colnames(super_task), value = T) ,
   fk = grep('fk_',colnames(super_task), value = T))# colnames(dplyr::select(super_task, fk_Flow_rate:fk_EC, -fk_O2)))

message("cmb_feature_groups done")


if(F){# if we want a small test set
  idx <- sample(nrow(grd), 3)
  grd_sample <- grd[idx, ]
  # bmr_a_sample <- bmr_a[idx, ]
  # bmr_sample <- bmr[idx]
} # if we want a small test set

grd_sample <- grd

message("grd_sample done with length ", nrow(grd_sample))

# if we want only pdp 
if(F){
  pdp_imp_mcl <- mclapply(1:nrow(grd_sample),function(i){
    task_tmp <- grd_sample[i, 'task'][[1]][[1]]$clone(deep = T)
    learner_tmp <- grd_sample[i, 'learner'][[1]][[1]]$clone(deep = T)
    task_feature_names <- task_tmp$feature_names
    pdp_lst <- list() # igasse kogume 10 hinnangut
    message("Processing item ", i)
    for(j in 1:10){
      message("Processing j ", j)
      learner_tmp$train(task_tmp)
      imp.pred <- iml::Predictor$new(model = learner_tmp, data = task_tmp$data())
      
      eff_lst <- list()
      for(e in 1:length(task_feature_names)){
        feature_range <- range(task_tmp$data(cols=task_feature_names[e]), na.rm=T)# get feature range
        feature_grid <- seq(feature_range[1], feature_range[2], length=20)
        eff_lst[[e]] <- iml::FeatureEffect$new(predictor = imp.pred, feature = task_feature_names[e], method = 'pdp', grid.points = feature_grid)$results %>% filter(.class == 1) } # end e loop
      
      names(eff_lst) <- task_feature_names
      tmp <- lapply(eff_lst, function(x){colnames(x)[c(1,3)] <- c('X','value'); return(dplyr::select(x, value, X))})
      eff_df <- bind_rows(tmp, .id = 'feature')
      pdp_lst[[j]] <- eff_df
    } # end j loop
    return(bind_rows(pdp_lst, .id = 'id') )
  }, mc.cores = 1)
  
  save(pdp_imp_mcl, file = './dat/pdp_imp_mcl.rda')
}

# if we want iml and ale
if(F){
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
}

# if we want 10 averaged iml and ale
# if we want iml and ale

if(T){
  
  all_imp_mcl <- mclapply(1:nrow(grd_sample),function(i){         # i loop: 1 to 3624
    task_tmp <- grd_sample[i, 'task'][[1]][[1]]$clone(deep = T)
    learner_tmp <- grd_sample[i, 'learner'][[1]][[1]]$clone(deep = T)
    if(grepl('_rg_', learner_tmp$id)){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
    feats = NULL # feats for permutational feature importance
    if(grepl('_cmb', task_tmp$id)){feats <- cmb_feature_groups}
    task_feature_names <- task_tmp$feature_names
    
    iml_lst <-  ale_lst <- list() # igasse kogume 10 hinnangut
    message("Processing item ", i)
    for(j in 1:10){
      message("Processing replicate j ", j)
      learner_tmp$train(task_tmp)
      imp.pred <- iml::Predictor$new(model = learner_tmp, data = task_tmp$data(), y = task_tmp$target_names)
      iml_lst[[j]] <- try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results %>% dplyr::select(feature, importance), silent = T)
      
      eff_lst <- list()
      for(e in 1:length(task_feature_names)){eff_lst[[e]] <- iml::FeatureEffect$new(predictor = imp.pred, feature = task_feature_names[e], grid.size = 60)$results %>% filter(.class ==1)}
      names(eff_lst) <- task_feature_names
      tmp <- lapply(eff_lst, function(x){colnames(x)[3:4] <- c('value','X'); return(dplyr::select(x, value, X))})
      eff_df <- bind_rows(tmp, .id = 'feature')
      ale_lst[[j]] <- eff_df
    } # end j loop
    return(list(bind_rows(iml_lst, .id = 'id') %>% summarise(., importanceSD = sd(importance, na.rm=T), importance = mean(importance, na.rm=T), .by=feature), bind_rows(ale_lst, .id = 'id') %>% summarise(., valueSD = sd(value, na.rm=T), value = mean(value, na.rm=T), .by=c(feature, X)) ))
  }, mc.cores = 10)

  
  
  save(all_imp_mcl, file = './dat/all_imp_mcl_v2.rda')
}
 

