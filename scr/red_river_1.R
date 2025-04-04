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
# rm(list=grep('reddir',ls(), value = T, invert = T)) # garbage collection all but reddir

# super_task ####

load(paste0(reddir, 'dat/super_task.rda') ) 
# dplyr::select(super_task, starts_with('fk_')) %>% cor(use = 'pairwise.complete.obs')
# vol2 specifics - constrain super_task
super_task <- dplyr::select(super_task, -c("fk_Flow_rate","fk_NH4","fk_O2","fk_COD"))


### tasks on the fly ####

if(T){
  #     tsk_cmb_BatD$truth() %>% table()
  
  BatD <- transmute(super_task, y = as.numeric(as.logical(Batrachospermum))) 
  RedD <- dplyr::select(super_task, Audouniella.hermannii:Lemanea.rigida) %>% transmute(y = as.numeric(as.logical(rowSums(.))))
  
  # cmb task
  tsk_cmb_BatD <- dplyr::select(super_task, fk_Flow_velocity:mk_soo) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_cmb')
  tsk_cmb_RedD <- dplyr::select(super_task, fk_Flow_velocity:mk_soo) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_cmb')
  
  # FK task
  tsk_fk_BatD <- dplyr::select(super_task, starts_with('fk_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_fk')
  tsk_fk_RedD <- dplyr::select(super_task, starts_with('fk_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_fk')
  
  # mk task
  tsk_mk_BatD <- dplyr::select(super_task, starts_with('mk_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_mk')
  tsk_mk_RedD <- dplyr::select(super_task, starts_with('mk_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_mk')
  # pk task
  tsk_pk_BatD <- dplyr::select(super_task, starts_with('pk_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_pk')
  tsk_pk_RedD <- dplyr::select(super_task, starts_with('pk_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'RedD_pk')
  
  # sete task
  tsk_st_BatD <- dplyr::select(super_task, starts_with('st_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_st')
  tsk_st_RedD <- dplyr::select(super_task, starts_with('st_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_st')
  
  # ap task
  tsk_ap_BatD <- dplyr::select(super_task, starts_with('ap_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_ap')
  tsk_ap_RedD <- dplyr::select(super_task, starts_with('ap_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_ap')
  
  tsks <- mget(ls(pattern = '^tsk_')) # 12 tasks
 
} # 12 tasks on the fly; tsks - task list of 12

tsks <- mget(ls(pattern = '^tsk_')) # 2 fk tasks

### learners on the fly ####

# Agnostic Learners on the fly # requires mlr3tuningspaces
if(T){
  # ranger tuning spaces 
  lrn_D_rg <- lts("classif.ranger.default", num.trees = to_tune(p_int(4, 2048, tags = "budget")))$get_learner(predict_type = 'prob', importance = 'impurity')
  lrn_1_rg <- lts("classif.ranger.rbv1", num.trees = to_tune(p_int(4, 2048, tags = "budget")))$get_learner(predict_type = 'prob', importance = 'impurity')
  lrn_2_rg <- lts("classif.ranger.rbv2", num.trees = to_tune(p_int(4, 2048, tags = "budget")))$get_learner(predict_type = 'prob', importance = 'impurity')
  
  # xgboost tuning spaces 
  lrn_D_xg <- lts("classif.xgboost.default", nrounds = to_tune(p_int(4, 2048, tags = "budget")))$get_learner(predict_type = 'prob')
  lrn_1_xg <- mlr3tuningspaces::lts("classif.xgboost.rbv1", booster = to_tune(p_fct(levels = c("gbtree", "dart"))), nrounds = to_tune(p_int(4, 2048, tags = "budget")))$get_learner(predict_type = 'prob')
  lrn_2_xg <- lts("classif.xgboost.rbv2", booster = to_tune(p_fct(levels = c("gbtree", "dart"))),nrounds = to_tune(p_int(4, 2048, tags = "budget")))$get_learner(predict_type = 'prob')
  
  # make it graph
  
  glrn_D_rg <- as_learner(po("imputemedian") %>>% lrn_D_rg)
  glrn_1_rg <- as_learner(po("imputemedian") %>>% lrn_1_rg)
  glrn_2_rg <- as_learner(po("imputemedian") %>>% lrn_2_rg)
  
  # xgboost graph tuning spaces 
  glrn_D_xg <- as_learner(po("imputemedian") %>>% lrn_D_xg)
  glrn_1_xg <- as_learner(po("imputemedian") %>>% lrn_1_xg)
  glrn_2_xg <- as_learner(po("imputemedian") %>>% lrn_2_xg)
  
} # 6 agnostic learners with tuning spaces on the fly; provides learner list: glrn_xx_lst
# make graph learner list
glrn_xx_lst <- mget(ls(pattern = '^glrn_._.g')) #

save(tsks, glrn_xx_lst, file = paste0(reddir, 'dat/red_river_1.rda'))
message("script 1 completed \n")

