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

library(parallel)

reddir <- system("find ~/Documents -name 'red_river.Rproj' ", intern = T) %>% sub('red_river.Rproj','', .)

super_task <- dplyr::select(super_task, -c("fk_Flow_rate","fk_NH4","fk_O2","fk_COD"))

cmb_feature_groups <- list(
  ap = grep('ap_',colnames(super_task), value = T) ,
  mk = grep('mk_',colnames(super_task), value = T) ,
  pk = grep('pk_',colnames(super_task), value = T) ,
  st = grep('st_',colnames(super_task), value = T) ,
  fk = grep('fk_',colnames(super_task), value = T))

message("cmb_feature_groups done")

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

save(all_imp_mcl, file = paste0(reddir, 'dat/all_imp_mcl_v2.rda'))
