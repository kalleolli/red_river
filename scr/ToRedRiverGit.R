

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


## `red_geol_features.R` -> `red_geol_features.rda` võtab jubetumalt aega
# `RedRiverIni.R` starts from excel files, saves darlings in `./dat/red_river_ini.rda`.


# `red_red.R` requires objects `/dat/red_river_ini.rda` from `RedRiverIni.R` and `/dat/red_geol_features.rda` from `Red_geol_features.R`

# 'red_tasks.rda' tehakse valmis skriptis 'red_red.R'
# `scr/red_hyptune16.R` võtab taskid `dat/red_tasks.rda`, teeb learnerid on-the-fly ja chunk `red Hyperband Tuning` teeb pika hyperband tuuningu.

reddir <- system("find ~/Documents -name 'red_river.Rproj' ", intern = T) %>% sub('red_river.Rproj','', .)

rm(list=grep('reddir',ls(), value = T, invert = T)) # garbage collection all but reddir

## super_task ####

load(paste0(reddir, 'dat/super_task.rda') ) 


### mlr3 tasks on the fly ####

if(T){
  #     tsk_cmb_BatD$truth() %>% table()
  
  BatD <- transmute(super_task, y = as.numeric(as.logical(Batrachospermum))) 
  RedD <- select(super_task, Audouniella.hermannii:Lemanea.rigida) %>% transmute(y = as.numeric(as.logical(rowSums(.))))
  
  # cmb task
  tsk_cmb_BatD <- dplyr::select(super_task, fk_Flow_rate:mk_soo, -fk_O2) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_cmb')
  tsk_cmb_RedD <- dplyr::select(super_task, fk_Flow_rate:mk_soo, -fk_O2) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_cmb')
  
  # FK task
  tsk_fk_BatD <- dplyr::select(super_task, starts_with('fk_'), -fk_O2) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_fk')
  tsk_fk_RedD <- dplyr::select(super_task, starts_with('fk_'), -fk_O2) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_fk')
  
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
  # tsks[[1]]
} # 12 tasks on the fly; tsks - task list of 12


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
  
  # make graph
  
  glrn_D_rg <- as_learner(po("imputemedian") %>>% lrn_D_rg)
  glrn_1_rg <- as_learner(po("imputemedian") %>>% lrn_1_rg)
  glrn_2_rg <- as_learner(po("imputemedian") %>>% lrn_2_rg)
  
  # xgboost graph tuning spaces 
  glrn_D_xg <- as_learner(po("imputemedian") %>>% lrn_D_xg)
  glrn_1_xg <- as_learner(po("imputemedian") %>>% lrn_1_xg)
  glrn_2_xg <- as_learner(po("imputemedian") %>>% lrn_2_xg)
  
  # ls(patter = 'lrn_') 12 # learners
} # agnostic learners with tuning spaces on the fly

# make graph learner list

glrn_xx_lst <- mget(ls(pattern = '^glrn_._.g')) #

# callr hyperband tuning ####

# saves hypertune headers to folder ./dat/callrdat/ 
library(callr)

if(F){ # takes time with callr
  ex2 <- expression(
    library(mlr3hyperband),
    library(mlr3tuning),
    library(mlr3verse),
    library(tidyverse),
    for(i in 1:length(tsks)){
      hyptune <-  parallel::mclapply(glrn_xx_lst, function(x){
        try(mlr3tuning::tune(
          tuner = mlr3tuning::tnr("hyperband", eta = 2, repetitions = 1), 
          task = tsks[[i]], 
          learner = x, # saame siia kõik 6 learnerit suruda
          resampling = mlr3::rsmp("cv", folds = 3),
          measures = mlr3::msr("classif.ce"),
          terminator = bbotk::trm("none")), silent = T )}, mc.cores = 6) # siit on vaja salvestada ainult headerid
      hyptune_heads <- lapply(lapply(hyptune, '[[', 'archive'), '[[', 'data') %>% lapply(., function(x){arrange(x, classif.ce) %>% head(50) %>% select(classif.ce, x_domain)})
      savedir <- paste0('./dat/hyptune_',names(tsks)[i],'.rda')
      save(hyptune_heads, file = savedir)})
  
  rp2 <- r_bg(function(ex2, glrn_xx_lst, tsks) eval(ex2), args = list(ex2, glrn_xx_lst, tsks),  stdout = "./dat/out.txt", stderr = "./dat/err.txt") 
  
  rp2$is_alive()
  
 
} # takes time with callr


# benchmark ###

if(T){
  # ranger does not accept missing data, we impute median values with graph learner
  glrn_fl <- as_learner(po("imputemedian") %>>% lrn("classif.featureless", method = 'weighted.sample', predict_type = 'prob'))
  glrn_xg <- as_learner(po("imputemedian") %>>% lrn("classif.xgboost", predict_type = 'prob'))
  glrn_rg <- as_learner(po("imputemedian") %>>% lrn("classif.ranger", predict_type = 'prob', importance = 'impurity'))
  
  
  # glrn_xg$param_set$values
  # glrn_rg$param_set$values
  
  # load hyptune header files
  # BatD_hyp_files <- list.files(paste0(reddir,'dat/callrdat/')) %>% grep('BatD.rda', .,  value = T)
  # RedD_hyp_files <- list.files(paste0(reddir,'dat/callrdat/')) %>% grep('RedD.rda', .,  value = T)
  
  hyp_files <- list.files(paste0(reddir,'dat/')) %>% grep('hyptune', .,  value = T) # 12
  
# make a list of learners with hyperband tuned parameters included  
  # list of 12 - each is combo of Red/Bat, x 6 feature groups (cmb, fk, ap, pk, mk, st)
  # each of 12 is 50 header x 6 learners = 300 learners per list slot
# loop over 12 hyp_files; load (as hyptune_heads)
  
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
    print(hf)
  }
  names(hyp_lst) <- sub('hyptune_tsk_','',hyp_files) %>% sub('.rda','',.)
 
  names(tsks) # from tasks_on_the_fly

  
  # benchmark grid - hyp_lst
  glrn_xg$id <- 'glrn_0_xg_0'
  glrn_rg$id <- 'glrn_0_rg_0'
  
  grd_lst <- list()
  for(i in 1:length(hyp_lst)){
    grd_lst[[i]] <- mlr3::benchmark_grid(
      tasks = tsks[i],
      learners = c(glrn_rg, glrn_xg, hyp_lst[[i]]),
      rsmp('cv', folds = 5))
  }
  # unlist
  grd <- bind_rows(grd_lst) # 3624  x  3 benchmark grid
  
  save(grd, file = './dat/grd.rda')
  library(future)
  plan(multisession) # speeds up benchmarking
  
  # makes or loads bmr and bmr_a
  if(file.exists(paste0(reddir, 'dat/bmr.rda'))){load(paste0(reddir, 'dat/bmr.rda'))} else 
  {system.time(bmr <- mlr3::benchmark(grd)) 
    bmr_a <- bmr$aggregate() ; dim(bmr_a)
    
    # parse bmr categories; delete capital D
    bmr_a <- strsplit(bmr_a$task_id, '_') %>% do.call(rbind, .) %>% as.data.table() %>% transmute(., red_id = sub('D','',V1), feat_id = V2) %>% bind_cols(bmr_a, .)
    # table(bmr_a$feat_id) # all 604
    bmr_a <- strsplit(bmr_a$learner_id, '_') %>% do.call(rbind, .) %>% as.data.table() %>% transmute(., hyp_id = V2, mod_id = sub('b','',V3), batch_id = V4) %>% bind_cols(bmr_a, .)
    save(bmr, bmr_a, file = paste0(reddir, 'dat/bmr.rda'))
    }

    # benchmark verdict: with un-tuned models, rg is much better than xg. However, rg does not improve with tuning, while xg does and finally beats rg on classif.ce front.
  
  if(T){
  ggplot(filter(bmr_a, red_id != 'Bat', hyp_id !=  '0')) + geom_boxplot(aes(x = mod_id, y = classif.ce, col = hyp_id)) +
    geom_hline(data = filter(bmr_a, red_id != 'Bat', hyp_id ==  '0'), mapping = aes(yintercept = classif.ce, colour = mod_id), linetype="dashed") +
    facet_grid(rows = vars(feat_id),  scales = 'free_y')
  
  ggplot(filter(bmr_a, red_id != 'Red', hyp_id !=  '0')) + geom_boxplot(aes(x = mod_id, y = classif.ce, col = hyp_id)) +
    geom_hline(data = filter(bmr_a, red_id != 'Bat', hyp_id ==  '0'), mapping = aes(yintercept = classif.ce, colour = mod_id), linetype="dashed") +
    facet_grid(rows = vars(feat_id),  scales = 'free_y')
} # ggplot benchmarks
 
  }
  

# both importance ####

# do it linearly - horribly long, but does not screw up.

if(T){  # cmb_feature_groups needed for iml
  cmb_feature_groups <- list(
    ap = grep('ap_',colnames(super_task), value = T) ,
    mk = grep('mk_',colnames(super_task), value = T) ,
    pk = grep('pk_',colnames(super_task), value = T) ,
    st = grep('st_',colnames(super_task), value = T) ,
    fk = colnames(select(super_task, fk_Flow_rate:fk_EC, -fk_O2)))
} # cmb_feature_groups needed for iml


if(F){ #  both_importance
  
  if(F){# if we want a small test set
    idx <- sample(nrow(grd), 10)
    grd_sample <- grd[idx, ]
    bmr_a_sample <- bmr_a[idx, ]
  } # if we want a small test set
  # if we want whole set:
  grd_sample <- grd
  bmr_a_sample <- bmr_a
  
  
  both_imp <- list() # initiate
  for(i in 1:nrow(grd_sample)){
    learner_tmp <- grd_sample[i,'learner'][[1]][[1]]$clone(deep = T)
    if(bmr_a_sample$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
    feats = NULL
    if(bmr_a_sample$feat_id[i] == 'cmb'){feats <- cmb_feature_groups}
    iml_lst <- mlr_lst <- list() # igasse kogume 10 hinnangut
    for(j in 1:10){
      print(j)
      learner_tmp$train(grd_sample[i,'task'][[1]][[1]])
      imp.pred <- iml::Predictor$new(model = learner_tmp, 
                                     data = grd_sample[i,'task'][[1]][[1]]$data(), 
                                     y = grd_sample[i,'task'][[1]][[1]]$target_names)
      mlr_lst[[j]] <- learner_tmp$base_learner()$importance()
      iml_lst[[j]] <- try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results %>% dplyr::select(feature, importance), silent = T)
    }
    both_imp[[i]] <- list(mlr_df <- bind_rows(mlr_lst, .id = 'id'), iml_df <- bind_rows(iml_lst, .id = 'id'))
    
    cat(i, class(iml_imp[[i]]),'\n')
  } # both importance linear
  
  # faster versions are lapply and mclapply
  
  system.time(bothl3_imp <- mclapply(1:nrow(grd_sample), function(i){
    learner_tmp <- grd_sample[i,'learner'][[1]][[1]]$clone(deep = T)
    if(bmr_a_sample$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
    feats = NULL
    if(bmr_a_sample$feat_id[i] == 'cmb'){feats <- cmb_feature_groups}
    iml_lst <- mlr_lst <- list() # igasse kogume 10 hinnangut
    for(j in 1:10){
      learner_tmp$train(grd_sample[i,'task'][[1]][[1]])
      imp.pred <- iml::Predictor$new(model = learner_tmp, 
                                     data = grd_sample[i,'task'][[1]][[1]]$data(), 
                                     y = grd_sample[i,'task'][[1]][[1]]$target_names)
      mlr_lst[[j]] <- learner_tmp$base_learner()$importance()
      iml_lst[[j]] <- try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results %>% dplyr::select(feature, importance), silent = T)
    }
    return(list(bind_rows(mlr_lst, .id = 'id'), bind_rows(iml_lst, .id = 'id')))
  }, mc.cores = 10)) # both importance lapply
  # lapply 1 tuum ca 350% 494 (= 50h) sek
  # mclapply 10, 258 sek (= 26 h), swap läks ikka 1.5Gb peale
  
  
  
  
  
  system.time(iml_imp_tmp <- lapply(1:nrow(grd_sample), function(i){
    learner_tmp <- grd_sample[i,'learner'][[1]][[1]]$clone(deep = T)
    if(bmr_a_sample$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
    learner_tmp$train(grd_sample[i,'task'][[1]][[1]])
    imp.pred <- iml::Predictor$new(model = learner_tmp, 
                                   data = grd_sample[i,'task'][[1]][[1]]$data(), 
                                   y = grd_sample[i,'task'][[1]][[1]]$target_names)
    feats = NULL
    if(bmr_a_sample$feat_id[i] == 'cmb'){feats <- cmb_feature_groups}
    # return(learner_tmp$base_learner()$importance())
    return(try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results  %>% dplyr::select(feature, importance), silent = T))
  })) # 5929.052
  sapply(iml_imp_tmp, class)
  
} # both importance


# Feature Importance figs  ####

if(T){
  
  # partition both_imp to mlr and iml parts
  mlr.imp <- lapply(bothl3_imp, '[[', 2)
  
  # scale mlr importance estimates with decostand, max and total
  tmp <- lapply(mlr.imp, function(x){mutate(x, .by = 'id', impmax = vegan::decostand(importance, method = 'max'), imprange = vegan::decostand(importance, method = 'range'))})
  
  mlr.importance <- list()
  # add all classificators for downstream filtering
  for(i in 1:length(tmp)){
    mlr.importance[[i]] <- bind_cols(tmp[[i]], bmr_a_sample[i, classif.ce:batch_id])
  }
  mlr.importance <- bind_rows(mlr.importance)
  
  
  # box mlr
  
  dat <- filter(mlr.importance, feat_id == 'fk')
  
  ggplot(dat) + geom_boxplot(aes(y = impmax, x=feature, col=hyp_id)) + facet_grid(rows = vars(mod_id),  cols = vars(red_id), scales = 'free_y')  + ggtitle("MLR") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + theme(legend.position="none")
  
  ggplot(dat) +  geom_boxplot(aes(y = imprange, x=feature, col=hyp_id)) + facet_grid(rows = vars(mod_id),  cols = vars(red_id), scales = 'free_y')  + ggtitle("MLR") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + theme(legend.position="none")
  
  
  # box iml
  
  iml.imp <- lapply(bothl3_imp, '[[', 1)
  
  tmp <- list()
  for(i in 1:length(iml.imp)){
    tmp[[i]] <- pivot_longer(iml.imp[[i]], cols = -id, names_to = 'feature', values_to = 'importance') %>% 
      bind_cols(., bmr_a_sample[i,]) %>% 
      mutate(., .by = 'id', impmax = vegan::decostand(importance, method = 'max'), imprange = vegan::decostand(importance, method = 'range'))
  }
  iml.importance <- bind_rows(tmp)
  
  iml.dat <- filter(iml.importance, feat_id == 'fk')
  
  ggplot(iml.dat) + geom_boxplot(aes(y = impmax, x=feature, col=hyp_id)) + facet_grid(rows = vars(mod_id),  cols = vars(red_id), scales = 'free_y')  + ggtitle("IML") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + theme(legend.position="none")
  
  ggplot(iml.dat) +  geom_boxplot(aes(y = imprange, x=feature, col=hyp_id)) + facet_grid(rows = vars(mod_id),  cols = vars(red_id), scales = 'free_y')  + ggtitle("IML") + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + theme(legend.position="none")
  
  
} # importance figs


# ALE ####

# ale effect needs trained learner again

if(F){# if we want a small test set
  idx <- sample(nrow(grd), 10)
  grd_sample <- grd[idx, ]
  bmr_a_sample <- bmr_a[idx, ]
} # if we want a small test set
# if we want whole set:
# grd_sample <- grd
# bmr_a_sample <- bmr_a



ale_eff <- list() # initiate as in both_imp
for(i in 1:nrow(grd_sample)){
  learner_tmp <- grd_sample[i,'learner'][[1]][[1]]$clone(deep = T)
  if(bmr_a_sample$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
  ale_lst <- list() # igasse kogume 10 hinnangut
    task_feature_names <- grd_sample[i,'task'][[1]][[1]]$feature_names
  for(j in 1:10){
    print(j)
    learner_tmp$train(grd_sample[i,'task'][[1]][[1]])
    imp.pred <- iml::Predictor$new(model = learner_tmp, data = grd_sample[i,'task'][[1]][[1]]$data(), y = grd_sample[i,'task'][[1]][[1]]$target_names)
   
    #iml_lst[[j]] <- try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results %>% dplyr::select(feature, importance), silent = T)
    
    eff_lst <- list()
    for(e in 1:length(task_feature_names)){eff_lst[[e]] <- iml::FeatureEffect$new(predictor = imp.pred, feature = task_feature_names[e], grid.size = 60)$results %>% filter(.class ==1)}
    names(eff_lst) <- task_feature_names
    tmp <- lapply(eff_lst, function(x){colnames(x)[3:4] <- c('value','X'); return(dplyr::select(x, value, X))})
    eff_df <- bind_rows(tmp, .id = 'feature')
    ale_lst[[j]] <- eff_df
  } # end j loop
  # both_imp[[i]] <- list(mlr_df <- bind_rows(mlr_lst, .id = 'id'), iml_df <- bind_rows(iml_lst, .id = 'id'))
    ale_eff[[i]] <- ale_lst
  cat(i, class(ale_eff[[i]]),'\n')
} # ALE linear

tmp <- lapply(ale_eff, bind_rows, .id ='batch') # list of 


# All_impact ####

## all impact linear ####
all_imp <- list() # initiate
system.time(for(i in 1:nrow(grd_sample)){
  learner_tmp <- grd_sample[i,'learner'][[1]][[1]]$clone(deep = T)
  if(bmr_a_sample$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
  feats = NULL
  if(bmr_a_sample$feat_id[i] == 'cmb'){feats <- cmb_feature_groups}
  task_feature_names <- grd_sample[i,'task'][[1]][[1]]$feature_names
  iml_lst <- mlr_lst <- ale_lst <- list() # igasse kogume 10 hinnangut
  for(j in 1:10){
    print(j)
    learner_tmp$train(grd_sample[i,'task'][[1]][[1]])
    imp.pred <- iml::Predictor$new(model = learner_tmp, data = grd_sample[i,'task'][[1]][[1]]$data(), y = grd_sample[i,'task'][[1]][[1]]$target_names)
    
    mlr_lst[[j]] <- learner_tmp$base_learner()$importance()
    iml_lst[[j]] <- try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results %>% dplyr::select(feature, importance), silent = T)
    
    eff_lst <- list()
    for(e in 1:length(task_feature_names)){eff_lst[[e]] <- iml::FeatureEffect$new(predictor = imp.pred, feature = task_feature_names[e], grid.size = 60)$results %>% filter(.class ==1)}
    names(eff_lst) <- task_feature_names
    tmp <- lapply(eff_lst, function(x){colnames(x)[3:4] <- c('value','X'); return(dplyr::select(x, value, X))})
    eff_df <- bind_rows(tmp, .id = 'feature')
    ale_lst[[j]] <- eff_df
  } # end j loop
  
  all_imp[[i]] <- list(bind_rows(mlr_lst, .id = 'id'), bind_rows(iml_lst, .id = 'id'), bind_rows(ale_lst, .id = 'batch'))
  
  cat(i, class(all_imp[[i]]),'\n')
}) # all impact linear
  # linear elapsed 681.031, user 2433.073 

## all impact lapply ####



# all_impacts requires: cmb_feature_groups, grd, bmr_a
# save(cmb_feature_groups, grd, bmr_a, file = './dat/all_impact_input.rda')
# load('./dat/all_impact_input.rda')


if(F){# if we want a small test set
  idx <- sample(nrow(grd), 10)
  grd_sample <- grd[idx, ]
  bmr_a_sample <- bmr_a[idx, ]
} # if we want a small test set


system.time(
  all_imp_mcl <- mclapply(1:nrow(grd_sample),function(i){
  learner_tmp <- grd_sample[i,'learner'][[1]][[1]]$clone(deep = T)
  if(bmr_a_sample$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
  feats = NULL
  if(bmr_a_sample$feat_id[i] == 'cmb'){feats <- cmb_feature_groups}
  task_feature_names <- grd_sample[i,'task'][[1]][[1]]$feature_names
  iml_lst <- mlr_lst <- ale_lst <- list() # igasse kogume 10 hinnangut
  message("Processing item ", i)
  for(j in 1:10){
    learner_tmp$train(grd_sample[i,'task'][[1]][[1]])
    imp.pred <- iml::Predictor$new(model = learner_tmp, data = grd_sample[i,'task'][[1]][[1]]$data(), y = grd_sample[i,'task'][[1]][[1]]$target_names)
    
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
) # all impact lapply

# lapply     400 = 39 h, user 1451
# mclapply10 152 = 12 h, user 389
# mclapply1  393 = xx h, user 1442
# linear     391, user 1441

tmp <- mclapply(1:10, function(i){i+2; cat(i,'\n'); return(42)})






if(T){
  system.time(mlr_imp <- parallel::mclapply(1:nrow(grd), function(i){
    learner_tmp <- grd[i,'learner'][[1]][[1]]$clone(deep = T)
    if(bmr_a$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
    learner_tmp$train(grd[i,'task'][[1]][[1]])
    return(learner_tmp$base_learner()$importance())
  }, mc.cores = 10)) # 
  
  if(F){ # meil on probleem
    sapply(mlr_imp, length) %>% table() # problem? sõnaga, kui muutujat ei võetagi analüüsi, siis talle olulisust ei omistata? ja see varieerub iteratsioonist iteratsiooni
    
    idx <- which(sapply(mlr_imp, length) == 0)
    idx36 <- which(sapply(mlr_imp, length) == 20)
    i <- idx[1]
    grd[i,]
    learner_tmp <- grd[i,'learner'][[1]][[1]]$clone(deep = T)
    learner_tmp$train(grd[i,'task'][[1]][[1]])
    learner_tmp$base_learner()$importance() %>% length()
    learner_tmp$model
    tmp <- learner_tmp$predict(grd[i,'task'][[1]][[1]])
    tmp$prob %>% summary() # põhimõtteliselt viskab 0.500001 tõenäosusega nulli
    tmp$confusion
    tmp$obs_loss()
    tmp$score()
    
    
    # 1.5.2 data.table for Beginners
    # dt = data.table(x = 1:6, y = rep(letters[1:3], each = 2))
    # dt[, mean(x), by = "y"]
    tmp <- bind_cols(bmr_a, mlr_imp)
    tmp <- bmr_a; tmp$iml <- mlr_imp
    tmp[, do.call(bind_rows, iml), by = 'feat_id']
    tmp[, mean(classif.ce), by = c('feat_id','red_id', 'hyp_id', 'mod_id')] # 96 categories
    
    
    head(tmp) %>% select(iml) %>% bind_rows()
    head(tmp) %>% select(iml) %>% do.call(bind_rows,.)
    
    
  } # meil on probleem
}

# iml_importance ####

if(T){
  # from red_analyse.R; requires learner, task
  #      imp.pred <- iml::Predictor$new(learner, data = tsk0$data(), y = tsk0$target_names) # learner is a trained model
  # try( iml::FeatureImp$new(imp.pred, loss = "ce")$results , silent = T)
  
  cmb_feature_groups <- list(
                               ap = grep('ap_',colnames(super_task), value = T) ,
                               mk = grep('mk_',colnames(super_task), value = T) ,
                               pk = grep('pk_',colnames(super_task), value = T) ,
                               st = grep('st_',colnames(super_task), value = T) ,
                               fk = colnames(select(super_task, fk_Flow_rate:fk_EC, -fk_O2)))
  
  
  if(F){ # test iml_imp
    idx <- sample(nrow(grd), 10)
    grd_sample <- grd[idx, ]
    bmr_a_sample <- bmr_a[idx, ]
    
    iml_imp <- list()
    
    for(i in 1:nrow(grd_sample)){
      learner_tmp <- grd_sample[i,'learner'][[1]][[1]]$clone(deep = T)
      if(bmr_a_sample$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
      learner_tmp$train(grd_sample[i,'task'][[1]][[1]])
      imp.pred <- iml::Predictor$new(model = learner_tmp, 
                                     data = grd_sample[i,'task'][[1]][[1]]$data(), 
                                     y = grd_sample[i,'task'][[1]][[1]]$target_names)
      feats = NULL
      if(bmr_a_sample$feat_id[i] == 'cmb'){feats <- cmb_feature_groups}
      # return(learner_tmp$base_learner()$importance())
      iml_imp[[i]] <- try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results %>% dplyr::select(feature, importance), silent = T)
      print(i)
    }
    
    sapply(iml_imp, class)
    
    
    
    
    
    system.time(iml_imp_tmp <- lapply(1:nrow(grd_sample), function(i){
      learner_tmp <- grd_sample[i,'learner'][[1]][[1]]$clone(deep = T)
      if(bmr_a_sample$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
      learner_tmp$train(grd_sample[i,'task'][[1]][[1]])
      imp.pred <- iml::Predictor$new(model = learner_tmp, 
                                     data = grd_sample[i,'task'][[1]][[1]]$data(), 
                                     y = grd_sample[i,'task'][[1]][[1]]$target_names)
      feats = NULL
      if(bmr_a_sample$feat_id[i] == 'cmb'){feats <- cmb_feature_groups}
      # return(learner_tmp$base_learner()$importance())
      return(try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results  %>% dplyr::select(feature, importance), silent = T))
    })) # 5929.052
    sapply(iml_imp_tmp, class)
    
    
    
    
  } # test iml_imp
  
  
  iml_imp <- list()
  
  
  for(i in 1:nrow(grd)){
    learner_tmp <- grd[i,'learner'][[1]][[1]]$clone(deep = T)
    if(bmr_a$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
    learner_tmp$train(grd[i,'task'][[1]][[1]])
    imp.pred <- iml::Predictor$new(model = learner_tmp, 
                                   data = grd[i,'task'][[1]][[1]]$data(), 
                                   y = grd[i,'task'][[1]][[1]]$target_names)
    feats = NULL
    if(bmr_a$feat_id[i] == 'cmb'){feats <- cmb_feature_groups}
    # return(learner_tmp$base_learner()$importance())
    iml_imp[[i]] <- try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results %>% dplyr::select(feature, importance), silent = T)
    cat(i, class(iml_imp[[i]]),'\n')
  }

    
  sapply(iml_imp, class)
  # save the darling
  
  save(iml_imp, file = paste0(reddir, 'dat/callrdat/importance.rda'))
  load(paste0(reddir, 'dat/callrdat/importance.rda'))
  
  
}

# parallel iml_importance ####

if(T){
  
system.time(iml_imp <- parallel::mclapply(1:nrow(grd_sample), function(i){
    learner_tmp <- grd[i,'learner'][[1]][[1]]$clone(deep = T)
    if(bmr_a$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
    learner_tmp$train(grd[i,'task'][[1]][[1]])
    imp.pred <- iml::Predictor$new(model = learner_tmp, 
                                   data = grd[i,'task'][[1]][[1]]$data(), 
                                   y = grd[i,'task'][[1]][[1]]$target_names)
    feats = NULL
    if(bmr_a$feat_id[i] == 'cmb'){feats <- cmb_feature_groups}
    # return(learner_tmp$base_learner()$importance())
    return(try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results, silent = T))
  }, mc.cores = 10)) # 5929.052
  
}

## parallel both_importance ####

if(T){
  both_importance <- parallel::mclapply(1:nrow(grd), function(i){
    learner_tmp <- grd[i,'learner'][[1]][[1]]$clone(deep = T)
    if(bmr_a$mod_id[i] == 'rg'){learner_tmp$param_set$values$classif.ranger.importance = 'impurity'}
    feats = NULL
    if(bmr_a$feat_id[i] == 'cmb'){feats <- cmb_feature_groups}
    iml_lst <- mlr_lst <- list() # igasse kogume 10 hinnangut
    for(j in 1:10){
      learner_tmp$train(grd[i,'task'][[1]][[1]])
      imp.pred <- iml::Predictor$new(model = learner_tmp, 
                                     data = grd[i,'task'][[1]][[1]]$data(), 
                                     y = grd[i,'task'][[1]][[1]]$target_names)
      mlr_lst[[j]] <- learner_tmp$base_learner()$importance()
      iml_lst[[j]] <- try(iml::FeatureImp$new(imp.pred, loss = "ce", features = feats)$results, silent = T)
    }
    return(list(mlr_df <- bind_rows(mlr_lst), iml_df <- bind_rows(iml_lst)))
   
  }, mc.cores = 10) 
 
  
} # both_importance

