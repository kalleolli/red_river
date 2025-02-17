

library(mlr3)
library(mlr3hyperband)
library(mlr3tuning)
library(mlr3learners)
library(mlr3pipelines)
library(iml)
library(ranger)
library(xgboost)
library(tidyverse)

library(mlr3tuningspaces)
library(mlr3extralearners)




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
  
# loop over 12 hyp_files; load (as hyptune_heads)
  
  hyp_lst <- list() # siia kogume 12 hyptune learnerite setti
  
  for(hf in 1:length(hyp_files)){
    load(paste0(reddir,'dat/', hyp_files[hf])) # loads hyptune_heads; length 6
    # tahaks saada 6st listi, iga 50 learneriga, mille hüperparameetrid on hyptune_heads x_domain
    tmp <- lapply(seq_along(hyptune_heads), function(i){
      nimi <- names(hyptune_heads)[i]
      if(grepl('_rg', nimi)) learner1 <- glrn_rg$clone(deep = T) else learner1 <- glrn_xg$clone(deep = T)
      lst <- lapply(1:nrow(hyptune_heads[[i]]), function(x){learner2 <- learner1$clone(deep = T); learner2$param_set$set_values(.values = hyptune_heads[[i]]$x_domain[[x]]);  learner2$id <- paste(nimi,x, sep = '_'); return(learner2)})
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
  } # 4.7 Mb object
  # unlist
  grd <- bind_rows(grd_lst) # 3624  x  3 benchmark grid
  
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
 
  # mlr importance ####
  
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
 
   
  # iml_importance ####
  
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
  
  
  
  
  learner_tmp <- grd[i,'learner'][[1]][[1]]$clone(deep = T)
  learner_tmp$train(grd[i,'task'][[1]][[1]])
  
  imp.pred <- iml::Predictor$new(model = learner_tmp, 
                                 data = grd[i,'task'][[1]][[1]]$data(), 
                                 y = grd[i,'task'][[1]][[1]]$target_names)
  
  iml::FeatureImp$new(imp.pred, loss = "ce")$results
  
  
  
  learner_tmp <- grd[1,'learner'][[1]][[1]]$clone(deep = T)
  learner_tmp$param_set$set_values(.values = grd[1,'learner'][[1]][[1]]$param_set$values)
  learner_tmp$param_set$values$classif.ranger.importance = 'impurity'
  learner_tmp$train(grd[1,'task'][[1]][[1]])
  learner_tmp$base_learner()$importance()
  
  
  
  
  

  # ale business ####
  
  
  
  
  lapply(hyptune_heads, '[[', 'x_domain') %>% lapply(., bind_rows) # list of 6 with hyperparameters in tables
  
  # tahaks saada 6st listi, iga 50 learneriga, mille hüperparameetrid on hyptune_heads x_domain
  lapply(hyptune_heads, function(x){print(attr(x,'names'))})
  
  lst <- list(a=3, b=NA, c = runif(10))
  lapply(lst, function(x){attributes(x)})
  
  lapply(seq_along(lst), function(i){nimi <- names(lst)[i]; return(n)}) #print(names(lst)[i]))

  tmp <- lapply(seq_along(hyptune_heads), function(i){
    nimi <- names(hyptune_heads)[i]
    if(grepl('_rg', nimi)) learner1 <- glrn_rg$clone(deep = T) else learner1 <- glrn_xg$clone(deep = T)
    # lst <- lapply(hyptune_heads[[i]]$x_domain, function(x){learner2 <- learner1$clone(deep = T); learner2$param_set$set_values(.values = x);  learner2$id <- nimi; return(learner2)})
    lst <- lapply(1:nrow(hyptune_heads[[i]]), function(x){learner2 <- learner1$clone(deep = T); learner2$param_set$set_values(.values = hyptune_heads[[i]]$x_domain[[x]]);  learner2$id <- paste(nimi,x, sep = '_'); return(learner2)})
    return(lst)
        })
  
#  untrained_xx <- lapply(seq_along(glrn_xx_lst),function(i){glrn_xx_lst[[i]]$id = paste(names(glrn_xx_lst)[i],'0', sep = '_'); return(glrn_xx_lst[[i]])})
  glrn_xg$id <- 'glrn_0_xg_0'
  glrn_rg$id <- 'glrn_0_rg_0'
  
  grd = mlr3::benchmark_grid(
    tasks = tsk_ap_BatD,
    learners = c(glrn_rg, glrn_xg, unlist(tmp)),
    rsmp('cv', folds = 5)
  )
  

  grd = mlr3::benchmark_grid(
    tasks = tsk_ap_BatD,
    learners = list(glrn_fl, glrn_rg, glrn_xg, lrn_xg, glrnh_xg),
    rsmp('cv', folds = 5)
  )
  
  system.time(bmr <- mlr3::benchmark(grd)) # 800 sek
  a <- bmr$aggregate() 
  
  b <- bmr$score()
  
  
  table(a$learner_id)
  
tmp[, mean(classif.ce), by = "id"]
  
  
}

if(T){
tsk_cmb_BatD <- dplyr::select(super_task, Batrachospermum, Flow_rate:mk_soo) %>% mutate(Batrachospermum = as.numeric(as.logical(Batrachospermum))) %>% as_task_classif(target = "Batrachospermum", positive = "1", id = 'BatD_cmb')
  tsk_cmb_BatD$truth() %>% table()
  
  # benchmark and ...
  # .. graph learners
  
  library(future)
  plan(multisession)
 
  tsk_ap_BatD <- dplyr::select(super_task, Batrachospermum, mk_pold:mk_soo) %>% mutate(Batrachospermum = factor(as.numeric(as.logical(Batrachospermum))), mk_soo = sample(mk_soo)) %>% as_task_classif(target = "Batrachospermum", positive = "1", id = 'BatD_ap')
  
  
  grd_ap = mlr3::benchmark_grid(
    tasks = tsk_ap_BatD,
    learners = list(lrn("classif.featureless", method = 'weighted.sample', predict_type = 'prob'), glrn_rg, glrn_xg),
    rsmp('cv', folds = 10)
  )
  bmr_ap = mlr3::benchmark(grd_ap)
  bmr_ap$aggregate()
  
  bmr_fk$aggregate()
  
    } # 
    
   # names(hyptune)
   
    # probably too optimistic
    head(arrange(hyptune[[1]]$archive$data, classif.ce), 50) %>% select(classif.ce) %>% summary() # 0.09324  
    head(arrange(hyptune[[2]]$archive$data, classif.ce), 50) %>% select(classif.ce) %>% summary() # 0.09055 
    head(arrange(hyptune[[3]]$archive$data, classif.ce), 50) %>% select(classif.ce) %>% summary() # 0.09317  
    # untrained 0.09579961; vb xgboost oli rohkem treenitav?
    # compare with untuned
    grd_fk = mlr3::benchmark_grid(
      tasks = tsk_fk_BatD,
      learners = list(glrn_fl, glrn_rg),
      rsmp('cv', folds = 5)
    )
   # glrn_fl, glrn_rg
mlr3::benchmark(grd_fk)$aggregate()

    
  
# return to benchmark


 
  # random: 125/(125+1138)
  tmp <- tsk_cmb_BatD$truth()
  table(tmp==sample(tmp))[1]/1263 # 0.17 featureless
  
  bmr_cmb$aggregate() %>% filter(learner_id == 'classif.featureless')
  bmr_cmb$aggregate() %>% filter(learner_id == 'imputemedian.classif.ranger')
  bmr_cmb$aggregate() %>% filter(learner_id == 'imputemedian.classif.xgboost')
  

}



# check data
if(F){
  hkese <- read.table(file = 'dat/detailsed_seireandmed-11.csv', sep = ';', header = T, dec = ",") # 93362 
  # low pH 
  filter(redCCM, p_h_proovivotul < 6) %>% as.data.table()
  filter(redh, veekogu_kkr %in% c('VEE1062300','VEE1002700')) %>% as.data.table()
  
  filter(hkese, Veekogu.KKR %in% c('VEE1062300','VEE1002700')) %>% as.data.table() %>% filter(Näitaja.nimetus == 'pH (proovivõtul)')
    
    
   
  
} # ja ongi pH alla 5 kahes kohas!


# history: lst failid on red_geol_features.R tehtud, aeganõudev:
if(F){
#  aluspohi <- read_sf(dsn = paste0(geo_dir, "./shp/Aluspohi400k_shp/AP400_Avamus.shp")) %>% st_transform(., st_crs(vlg3)); st_agr(aluspohi) = "constant"
#  aluspohi$idx1 <- aluspohi$Indeks %>% str_sub(., 1, 1)
  
#  aluspohi.lst <- mclapply(geodb_sf$veekogu_kkr, function(x){
#    vg <- filter(geodb_sf, veekogu_kkr == x) %>% dplyr::select(geometry); 
#    tmp <- st_intersection(dplyr::select(aluspohi, idx1), vg); 
#    return(data.frame(idx1=tmp$idx1, area = as.numeric(st_area(tmp))) %>% dplyr::summarize(sarea = sum(area), .by = idx1))}, mc.cores = 10) # return 734
  # aluspohi.df <- bind_rows(aluspohi.lst, .id = 'id') %>%  pivot_wider(., names_from = idx1, values_from = sarea); dim(aluspohi.df) # 734 x 8
  
} # 734 unikaalset seirekoha valgala

# geol.lst to features ####
# code from red_geol_features.R


load(file = paste0(reddir,'dat/red_river_ini.rda')) # dat, red, redh, redCCM # from RedRiverIni.R

load(file = paste0(reddir, 'dat/red_geol_features.rda')) # geodb, ap, pk, mk, pinnakate_kood; from Red_geol_features.R
load(file = paste0(reddir, 'dat/red_tasks.rda'))

# kõik mis meil vaja: redCCM [1263 x 35] ja tsk_cmb_raw [1263 x  37]
tsk_BatD_ap_raw$data() %>% dim()


