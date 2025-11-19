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
# library(tidyverse)


reddir <- list.files(path = '~/Documents', full.names = TRUE, recursive = TRUE, pattern = 'red_river.Rproj') %>% dirname()
# rm(list=grep('reddir',ls(), value = T, invert = T)) # garbage collection all but reddir

# super_task ####

load(paste0(reddir, '/dat0/super_task.rda') ) # loads super_task
# super_task is a 1263 x 51 data.frame

if(F){
  str(super_task)
  # date - sampling date
  # veekogu_kkr - the code of the river
  # seirekoha_kkr - the code of the sampling site
  # X and Y - sampling site coordinates in Estonian Coordinate System of 1997 - EPSG:3301
  # Audouniella.hermannii Lemanea.rigida - red algae species abundance classes (0 - absent to 5 - abundant)
  # fk_Flow_rate to fk_EC - physical-chemical features (fk_ prefix)
  # st_gravel to st_limestone - substrate type features (st_ prefix)
  # ap_D to ap_S - watershed bedrock features (ap_ prefix) D- devon, O - ordovician + cambrian, S - silurian
  # pk_303 to pk_305 - watershed post-glacial sediments features (pk_ prefix)
  # mk_pold to mk_soo - watershed land cover features (mk_ prefix) pold - arable land, soo - bog, mets - forest, rohumaa - grassland
} # describe super_task data frame

# correlation matrix of physical-chemical features
# dplyr::select(super_task, starts_with('fk_')) %>% cor(use = 'pairwise.complete.obs')

# constrain super_task by dropping highly correlated features
super_task <- dplyr::select(super_task, -c("fk_Flow_rate","fk_NH4","fk_O2","fk_COD"))


### tasks on the fly ####

# make classification tasks for Batrachospermum and Red algae presence/absence
if(T){
  #     tsk_cmb_BatD$truth() %>% table()
  # presence/absence of Batrachospermum and all red algae
  BatD <- transmute(super_task, y = as.numeric(as.logical(Batrachospermum))) 
  RedD <- dplyr::select(super_task, Audouniella.hermannii:Lemanea.rigida) %>% transmute(y = as.numeric(as.logical(rowSums(.))))
  
  # cmb - classification tasks with all features
  tsk_cmb_BatD <- dplyr::select(super_task, fk_Flow_velocity:mk_soo) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_cmb')
  tsk_cmb_RedD <- dplyr::select(super_task, fk_Flow_velocity:mk_soo) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_cmb')
  
  # FK- classification tasks physical-chemical features
  tsk_fk_BatD <- dplyr::select(super_task, starts_with('fk_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_fk')
  tsk_fk_RedD <- dplyr::select(super_task, starts_with('fk_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_fk')
  
  # mk - classification tasks with watershed land-use features
  tsk_mk_BatD <- dplyr::select(super_task, starts_with('mk_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_mk')
  tsk_mk_RedD <- dplyr::select(super_task, starts_with('mk_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_mk')
  
  # pk - classification tasks with watershed post-glacial sediments features
  tsk_pk_BatD <- dplyr::select(super_task, starts_with('pk_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_pk')
  tsk_pk_RedD <- dplyr::select(super_task, starts_with('pk_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'RedD_pk')
  
  # st -  classification tasks with watershed substrate type features
  tsk_st_BatD <- dplyr::select(super_task, starts_with('st_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_st')
  tsk_st_RedD <- dplyr::select(super_task, starts_with('st_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_st')
  
  # ap -  classification tasks with watershed bedrock type features
  tsk_ap_BatD <- dplyr::select(super_task, starts_with('ap_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_ap')
  tsk_ap_RedD <- dplyr::select(super_task, starts_with('ap_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_ap')
  
  tsks <- mget(ls(pattern = '^tsk_')) # 12 tasks
 
} # 12 tasks on the fly; provides tsks - task list of 12 tasks: cmb, fk, mk, pk, st, ap x BatD, RedD


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
  
  # make graph learners with median imputation
  
  glrn_D_rg <- as_learner(po("imputemedian") %>>% lrn_D_rg)
  glrn_1_rg <- as_learner(po("imputemedian") %>>% lrn_1_rg)
  glrn_2_rg <- as_learner(po("imputemedian") %>>% lrn_2_rg)
  
  # xgboost graph tuning spaces 
  glrn_D_xg <- as_learner(po("imputemedian") %>>% lrn_D_xg)
  glrn_1_xg <- as_learner(po("imputemedian") %>>% lrn_1_xg)
  glrn_2_xg <- as_learner(po("imputemedian") %>>% lrn_2_xg)
  
} # 6 agnostic learners with tuning spaces on the fly; provides graph learner list: glrn_xx_lst
# make graph learner list
glrn_xx_lst <- mget(ls(pattern = '^glrn_._.g')) #

# save the darlings ####

if(!file.exists(file = paste0(reddir, '/dat/red_river_1.rda'))){
  save(tsks, glrn_xx_lst, file = paste0(reddir, '/dat/red_river_1.rda'))
  }

message("script 1 completed \n")
message(" proceed with red_river_2.R \n")

