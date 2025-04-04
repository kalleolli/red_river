
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

# source('./scr/ToRedRiverGit_v2.R')

## `red_geol_features.R` -> `red_geol_features.rda` võtab jubetumalt aega
# `RedRiverIni.R` starts from excel files, saves darlings in `./dat/red_river_ini.rda`.

# `red_red.R` requires objects `/dat/red_river_ini.rda` from `RedRiverIni.R` and `/dat/red_geol_features.rda` from `Red_geol_features.R`

# 'red_tasks.rda' tehakse valmis skriptis 'red_red.R'
# `scr/red_hyptune16.R` võtab taskid `dat/red_tasks.rda`, teeb learnerid on-the-fly ja chunk `red Hyperband Tuning` teeb pika hyperband tuuningu.

reddir <- system("find ~/Documents -name 'red_river.Rproj' ", intern = T) %>% sub('red_river.Rproj','', .)

rm(list=grep('reddir',ls(), value = T, invert = T)) # garbage collection all but reddir

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
  # tsks[[1]]
} # 12 tasks on the fly; tsks - task list of 12

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
  
  # ls(patter = 'lrn_') 12 # learners
  
  
  # make graph learner list
  glrn_xx_lst <- mget(ls(pattern = '^glrn_._.g')) #
  
} # 6 agnostic learners with tuning spaces on the fly; provides learner list: glrn_xx_lst

# HYPERBAND TUNING  ####


if(F){ # takes time with callr
  library(callr)  
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
      savedir <- paste0('./dat/hyptune2_',names(tsks)[i],'.rda')
      save(hyptune_heads, file = savedir)})
  
  rp2 <- r_bg(function(ex2, glrn_xx_lst, tsks) eval(ex2), args = list(ex2, glrn_xx_lst, tsks),  stdout = "./dat/out.txt", stderr = "./dat/err.txt") 
  
  rp2$is_alive()
  
} # takes time with callr

message("Start hypertune optimisation \n\n ")

tsks <- mget(ls(pattern = '^tsk_fk_')) # 2 fk tasks

if(T){
 
  system.time(for(i in 2:length(tsks)){ # 6 CPUs but each takes GB's of memory and swap
    hyptune <-  parallel::mclapply(glrn_xx_lst, function(x){
      try(mlr3tuning::tune(
        tuner = mlr3tuning::tnr("hyperband", eta = 2, repetitions = 1), 
        task = tsks[[i]], 
        learner = x, # saame siia kõik 6 learnerit suruda
        resampling = mlr3::rsmp("cv", folds = 3),
        measures = mlr3::msr("classif.ce"),
        terminator = bbotk::trm("none")), silent = T )}, mc.cores = 6) # siit on vaja salvestada ainult headerid
    hyptune_heads <- lapply(lapply(hyptune, '[[', 'archive'), '[[', 'data') %>% lapply(., function(x){arrange(x, classif.ce) %>% head(50) %>% dplyr::select(classif.ce, x_domain)})
    savedir <- paste0('./dat/hyptune_fk9_',names(tsks)[i],'.rda')
    save(hyptune_heads, file = savedir)} )
  message("Saved item ", savedir, "\n\n")
  
}


# provides 12 hyptune2_tsk_*.rda header files in ./dat folder


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
    print(hf)
  }
  names(hyp_lst) <- sub('hyptune2_tsk_','', hyp_files) %>% sub('.rda','',.)
 
} # provides hyp_lst - hyperparameter loaded learners, ready to be benchmarked


message("Start making a list of of benchmark grids  \n\n ")

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
  }
  # unlist
  grd <- bind_rows(grd_lst) # 3624  x  3 benchmark grid
  save(grd, file = './dat/grd2.rda') 
  
} # provides and saves 3624  x  3 ./dat/grd2.rda - benchmark grid object

message("Benchmark grid saved in ./dat/grd2.rda  \n\n ")

message("actual benchmarking with future speedup \n\n ")


library(future)
plan(multisession) # speeds up benchmarking

if(T){ # TAKES
  if(file.exists(paste0(reddir, 'dat/bmr2.rda'))){load(paste0(reddir, 'dat/bmr2.rda'))} else  {bmr <- mlr3::benchmark(grd);
    bmr_a <- bmr$aggregate();
    # parse bmr categories; delete capital D
    bmr_a <- strsplit(bmr_a$task_id, '_') %>% do.call(rbind, .) %>% as.data.table() %>% transmute(., red_id = sub('D','',V1), feat_id = V2) %>% bind_cols(bmr_a, .);
    # table(bmr_a$feat_id) # all 604
    bmr_a <- strsplit(bmr_a$learner_id, '_') %>% do.call(rbind, .) %>% as.data.table() %>% transmute(., hyp_id = V2, mod_id = sub('b','',V3), batch_id = V4) %>% bind_cols(bmr_a, .);
    save(bmr, bmr_a, file = paste0(reddir, 'dat/bmr2.rda'))
  }
  
} # provides (or loads) bmr and bmr_a; benchmarked hyperband tuned learners
 
# benchmark table 1 #### 
# Table 1

if(F){
  
  factor(iml.dat$red_id, levels = c('Bat','Red'), labels = c('Batrachospermum', 'all red algae')) 
  
  bench_tbl <- filter(bmr_a, batch_id != '0')  %>% 
    summarise(classif.ce = median(classif.ce), .by = c('red_id', 'feat_id','mod_id')) %>% 
    mutate(tuned = 1) %>% 
    bind_rows(., select(filter(bmr_a, batch_id == '0'), red_id, feat_id, mod_id, classif.ce) %>%
                mutate(tuned = 0)) %>% 
    mutate(feat_id = factor(feat_id, levels = c('cmb','fk','st', 'mk', 'pk', 'ap'), labels = c('all features','hydro chem','substrate','land use','land cover','bedrock'), ordered = T)) %>% 
    pivot_wider(., names_from = c(red_id, mod_id), values_from = classif.ce) %>%
    arrange(feat_id, tuned)

# add featureless loss function
  
design_combo = mlr3::benchmark_grid(
  tasks = tsks, # be very specific
  learners = list(as_learner(po("imputemedian") %>>% lrn("classif.featureless", method = 'weighted.sample'))),
  rsmp('loo')# loo = leave one out
)
bmr_combo = mlr3::benchmark(design_combo)  # actual benchmarking
(tmp <- bmr_combo$aggregate() %>% as.data.table())
filter(tmp, grepl('featureless',learner_id)) %>% arrange(task_id)

filter(tmp, grepl('featureless',learner_id)) %>% arrange(task_id)

} # table 1 benchmark
 


# Table 3 red algae x chem ####

# required super_task
if(F){
# helper func, if we want range, use probs c(0., 0.5, 1)
quantile_df <- function(x, probs = c(0.5, 0.05, 0.95)) {
    tibble(val = quantile(x, probs, na.rm = TRUE)) }
  
table3 <- list()

table3[['Audouniella']] <- 
  super_task %>% filter( if_any(starts_with('Audouniella') , ~ . > 0) ) %>% 
  reframe(across(starts_with('fk_'), quantile_df, .unpack = TRUE))

table3[['Batrachospermum']] <- 
  super_task %>% filter( if_any(starts_with('Batrachospermum') , ~ . > 0) ) %>% 
  reframe(across(starts_with('fk_'), quantile_df, .unpack = TRUE))

table3[['Lemanea']] <- 
  super_task %>% filter( if_any(starts_with('Lemanea') , ~ . > 0) ) %>% 
    reframe(across(starts_with('fk_'), quantile_df, .unpack = TRUE))

table3[['Hildenbrandia']] <- 
    super_task %>% filter( if_any(starts_with('Hildenbrandia') , ~ . > 0) ) %>% 
    reframe(across(starts_with('fk_'), quantile_df, .unpack = TRUE))
  
  lapply(table3, signif, 3) %>% lapply(select, -fk_O2_val)
  bind_rows(, .id='taxon') %>% select(-taxon) %>% t() %>% format(scientific=F)
  
  
  
  table3[[1]] %>% signif(.,3) %>% apply(2,function(y){paste0(y[1],' (',y[2],'-',y[3],')')})
  
  
  
  
  tmp <- lapply(table3, signif, 3) %>% lapply(select, -fk_O2_val) %>% lapply(function(x){apply(x,2, function(y){paste0(y[1],' (',y[2],'-',y[3],')')})} ) %>% bind_rows() %>% t()
  rownames(tmp) <- rownames(tmp) %>% sub('fk_','',.) %>% sub('_val','',.)
  tmp
  }

# benchmark verdict: with un-tuned models, rg performs better than xg. However, rg does not improve with tuning, while xg does and finally beats rg on classif.ce front.
# BAT ap - tuning does not beat untuned


# Feature Impacts ####

message("Running script all_impact_console \n\n ")
## do it in terminal:
system("R CMD BATCH ./scr/all_impact_console.R all_impact_out.txt")
# source('./scr/all_impact_console.R')


# fk variable selection ####

if(F){ # feature selection/elimination
  # https://mlr3book.mlr-org.com/chapters/chapter6/feature_selection.html
  # https://mlr-org.com/gallery/optimization/2023-02-07-recursive-feature-elimination/
  # fs , fsi (feature selection instances?), fselect, auto_fselector ( to be passed to benchmark or resample)
  # fs -  The FSelector Class: 
  #  fs("random_search") # trying random feature subsets until termination
  # fs("exhaustive_search") # trying all possible feature subsets
  # fs("sequential") # sequential forward or backward selection 
  # fs("rfe") # uses a learner’s importance scores to iteratively remove features with low feature importance
  # fs("design_points") #  trying all user-supplied feature sets
  # fs("genetic_search") # implementing a genetic algorithm which treats the features as a binary sequence and tries to find the best subset with mutations
  # fs("shadow_variable_search") # adds permuted copies of all features (shadow variables), performs forward selection, and stops when a shadow variable is selected 
} # feature selection shorts

if(F){
  # 6.1. filters

  library(mlr3filters)
  flts() # DictionaryFilters
  flt_gain = flt("information_gain")
  tsk_pen = tsk("penguins")
  flt_gain$calculate(tsk_pen)
  as.data.table(flt_gain)
  
  # one-liner
  flt("information_gain")$calculate(tsk('penguins')) %>% as.data.table()
  flt("importance")$calculate(tsk('penguins')) %>% as.data.table() # all zeros
  flt("permutation")$calculate(tsk('penguins')) %>% as.data.table() # takes 
  
  flts() 
  
  flt("information_gain")$calculate(tsk_fk_BatD) %>% as.data.table()  # BOD, Depth, Temp, 
  flt("permutation")$calculate(tsk_fk_RedD) %>% as.data.table() # pH, EC, Flow_velocity, Depth, O2, TP, Temp, BOD, TN - makes no sense
  
  flt("importance", learner = lrn('classif.xgboost') )$calculate(tsk_fk_RedD) %>% as.data.table()
  flt("importance", learner = lrn('classif.xgboost') )$calculate(tsk_fk_BatD) %>% as.data.table()
  
  
  
  # FK task
  #  If only a single filter method is to be used, the authors recommend to use a feature importance filter using random forest permutation importance
  
  load(paste0(reddir, 'dat/super_task.rda') ) 
  
  tsk_fk_BatD <- dplyr::select(super_task, starts_with('fk_')) %>% bind_cols(BatD,.) %>% as_task_classif(target = "y", positive = "1", id = 'BatD_fk')
  tsk_fk_RedD <- dplyr::select(super_task, starts_with('fk_')) %>% bind_cols(RedD,.) %>% as_task_classif(target = "y", positive = "1", id = 'Red_fk')
  
  
  lrn("classif.ranger")$param_set$levels$importance # to choose from "impurity"           "impurity_corrected" "permutation" 
  
  flt("importance", learner = as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity') ))$calculate(tsk_fk_BatD) %>% as.data.table()
  flt("importance", learner = as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity') ))$calculate(tsk_fk_RedD) %>% as.data.table()
  
  flt("permutation", learner = as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity') ))$calculate(tsk_fk_RedD) %>% as.data.table()
  
  
  flt("auc")$calculate(tsk_mk_RedD) %>% as.data.table()
  flt("importance", learner = as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity') ))$calculate(tsk_mk_RedD) %>% as.data.table()
  
  if(F){ # compare mlr & iml
    # vajas trained learnerit learner_tmp ja taski tsk_tmp
    
    learner_tmp <- as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity'))
    task_tmp <- tsk_fk_RedD
    flt("importance", learner = learner_tmp)$calculate(task_tmp) %>% as.data.table()
    learner_tmp$train(tsk_fk_RedD)
    
    imp.pred <- iml::Predictor$new(model = learner_tmp, data = task_tmp$data(), y = task_tmp$target_names)
    tmp_iml <- iml::FeatureImp$new(imp.pred, loss = "ce", features = NULL)$results %>% dplyr::select(feature, importance)
    
    # mlr importance
    learner_tmp$base_learner()$importance()
    
    # flt and mlr are similar. iml is different
    learner_tmp$base_learner()$importance()
    flt("importance", learner = learner_tmp)$calculate(task_tmp) %>% as.data.table()
    
    system.time(flt_imp <- mclapply(1:1000, function(x){flt("importance", learner = learner_tmp)$calculate(task_tmp) %>% as.data.table()}, mc.cores = 10)) # pressures memory 211 sek
    flt_df <- bind_rows(flt_imp)
   
    system.time(mlr_imp <- mclapply(1:1000, function(x){learner_tmp$train(tsk_fk_RedD); return(learner_tmp$base_learner()$importance())}, mc.cores = 10)) # memory pressure 128 sek
    mlr_df <- lapply(mlr_imp, function(x){tibble(feature = names(x), score = x)}) %>% bind_rows()
    
    
    
    ggplot(flt_df, aes(score, fill = feature, colour = feature)) + geom_density(alpha = 0.1) # density image
    
    ggplot(flt_df, aes(y=score, x=feature, fill = feature, colour = feature)) + geom_boxplot(alpha = 0.4) +  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none")# boxplot image
    ggplot(mlr_df, aes(y=score, x=feature, fill = feature, colour = feature)) + geom_boxplot(alpha = 0.4) +  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none")
    # verdict: mlr and flt are identical within randomisation
    
    
    system.time(tmp_imp <- mclapply(1:10, function(x){learner_tmp$train(tsk_fk_RedD); return(learner_tmp$base_learner()$importance())}, mc.cores = 10)) # memory pressure 128 sek
    tmp_df <- lapply(tmp_imp, function(x){tibble(feature = names(x), score = x)}) %>% bind_rows()
    
    ggplot(tmp_df, aes(y=score, x=feature, fill = feature, colour = feature)) + geom_boxplot(alpha = 0.4) +  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none")
    
    
    
    
    tmp <- tsk_fk_BatD$clone(deep = T)$select(setdiff(tsk_fk_RedD$feature_names, c('fk_O2','fk_Flow_rate','fk_COD','fk_NH4','fk_NO3')))
 
    tmp_df <- mclapply(1:10, function(x){learner_tmp$train(tmp); return(learner_tmp$base_learner()$importance())}, mc.cores = 10) %>%  lapply(., function(x){tibble(feature = names(x), score = x)}) %>% bind_rows()
    
 
    ggplot(tmp_df, aes(y=score, x=feature, fill = feature, colour = feature)) + geom_boxplot(alpha = 0.4) +  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none")
    
    
    learner_rf <- as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity'))
    learner_xg <-  lrn('classif.xgboost')
    
    no_feature <- c('fk_O2','fk_Flow_rate','fk_COD','fk_NH4', 'fk_TN')
    
    tsk_Bat <- tsk_fk_BatD$clone(deep = T)$select(setdiff(tsk_fk_BatD$feature_names, no_feature))
    tsk_Red <- tsk_fk_RedD$clone(deep = T)$select(setdiff(tsk_fk_RedD$feature_names, no_feature))
    
    Red_df <- mclapply(1:10, function(x){learner_rf$train(tsk_Red); return(learner_rf$base_learner()$importance())}, mc.cores = 10) %>%  lapply(., function(x){tibble(feature = names(x), score = x)}) %>% bind_rows() %>% mutate(y = 'red')
    Bat_df <- mclapply(1:10, function(x){learner_rf$train(tsk_Bat); return(learner_rf$base_learner()$importance())}, mc.cores = 10) %>%  lapply(., function(x){tibble(feature = names(x), score = x)}) %>% bind_rows() %>% mutate(y = 'bat')
   
    ggplot(bind_rows(Red_df, Bat_df), aes(y=score, x=feature, fill = feature, colour = feature)) + geom_boxplot(alpha = 0.4) +  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none") + facet_wrap(facets = 'y', ncol =1, scales = 'free_y')
    
    
    Red_df <- mclapply(1:10, function(x){learner_xg$train(tsk_Red); return(learner_xg$base_learner()$importance())}, mc.cores = 10) %>%  lapply(., function(x){tibble(feature = names(x), score = x)}) %>% bind_rows() %>% mutate(y = 'red')
    Bat_df <- mclapply(1:10, function(x){learner_xg$train(tsk_Bat); return(learner_xg$base_learner()$importance())}, mc.cores = 10) %>%  lapply(., function(x){tibble(feature = names(x), score = x)}) %>% bind_rows() %>% mutate(y = 'bat')
    
    ggplot(bind_rows(Red_df, Bat_df), aes(y=score, x=feature, fill = feature, colour = feature)) + geom_boxplot(alpha = 0.4) +  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none") + facet_wrap(facets = 'y', ncol =1, scales = 'free_y')
    
    # verdict:
    # xg san_NO3 red_TP laes (1st),           bat_TP laes (2nd)
    # rg san_NO3 red_TP laes (2nd),           bat_TP põhjas (last)
    
    # xg     NO3 red_TP laes (2nd),               bat_TP põhjas (last)
    # rg     NO3 red_TP põhjas,                   bat_TP põhjas (last)
    
    # xg san_TN red_TP laes (2nd),            bat_TP põhjas (last)
    # rg san_TN red_TP põhjas (last),         bat_TP põhjas (last)
    
    dplyr::select(super_task, starts_with('fk_'), -c('fk_O2','fk_Flow_rate','fk_COD','fk_NH4')) %>% cor(use = 'pairwise.complete.obs')#  > 0.5
    # TN-NO3 .857; lg 0.73
    # pH-O2  .52 ## log .47
    # TP-TN .446; log scale .25
    # Temp-NO3 -.445 ## log -.45
    summary(super_task$fk_NO3)
    ggplot(super_task, aes(x = log(fk_Temp), y=log(fk_NO3))) + geom_point()
    with(super_task, cor(log(fk_NO3), log(fk_Temp), use = 'pairwise.complete.obs'))
    
    ggplot(super_task, aes(x = (fk_pH), y=log(fk_O2p))) + geom_point()
    with(super_task, cor((fk_pH), (fk_O2p), use = 'pairwise.complete.obs'))
    
    tmp <- filter(super_task, fk_NO3 > 2, fk_Temp < 15) %>% dplyr::select(veekogu_kkr, X:Lemanea.rigida ) %>% unique() # 9 jõge
    tmp <- filter(super_task, fk_O2p > 100, fk_pH > 8) %>% dplyr::select(veekogu_kkr, X:Lemanea.rigida ) %>% unique()
    tmp$red <- dplyr::select(tmp, Audouniella.hermannii:Lemanea.rigida) %>% rowSums() %>% as.logical() %>% as.numeric()
    
    load(paste0(reddir, './dat/est.rda')) # est 40 kB
    load(paste0(reddir, './dat/red_rivers.rda')) # sampled rivers sf
    
    O2pH <- ggplot(data = est) +
      geom_sf(fill= "whitesmoke") + 
      geom_sf(data = filter(red_rivers, kr_kood %in% tmp$veekogu_kkr), aes(geometry = geometry), colour = 'gray', lwd = 0.1) + geom_point(tmp, mapping=aes(x=X, y=Y, col = factor(red))) # yesss! Pandivere nitraaditundlik ala. nitrate rich and cold rivers
# O2pH on hajusalt igal pool, no value. nothing to save
    ggsave(nitraaditundlik, file = paste0(reddir, './figs/nitraaditundlik.pdf'))
    
    if(F){ # Convert Named Character Vector to data.frame 
      mlr_imp[[1]] %>% as.list() %>% data.frame() # wide format one-row table
      mlr_imp[[1]] %>% as.list() %>% as_tibble()
      mlr_imp[[1]] %>% bind_rows()
      
      mlr_imp[[1]] %>% tibble::enframe() # long format; much better
      stack(mlr_imp[[1]])
      mlr_imp[[1]] %>% data.frame(keyName=names(.), value=., row.names=NULL)
      mlr_imp[[1]] %>% as_tibble(., rownames="feature")
      mlr_imp[[1]] %>% tibble(feature = names(.), score = .) # best, can name both cols
    } # Convert Named Character Vector to data.frame
    
  } # compare mlr & iml
  
  
  # flt is a sugar of mlr_filters
  flt('information_gain')$calculate(tsk('penguins')) %>% as.data.table()
  flt('importance')$calculate(tsk('penguins')) %>% as.data.table() # zeros, bc we need to supply learner with importance method
  flt('importance', learner = as_learner(po("imputemedian") %>>% lrn('classif.ranger')))$calculate(tsk('penguins')) %>% as.data.table() 
  # variable importance filters are incorporated in learners:
  ## [1] "classif.featureless" "classif.ranger"      "classif.rpart"      
  ## [4] "classif.xgboost"     "regr.featureless"    "regr.ranger"        
  ## [7] "regr.rpart"          "regr.xgboost"
  
  grd[4,]$learner[[1]] # tuned learners in grd
  grd[4,]$task[[1]] # for specific task 
  
  tskBat = tsk_fk_BatD$clone(deep = T)
  tskRed = tsk_fk_RedD$clone(deep = T)
  
  flt("importance", learner = as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity') ))$calculate(tskBat$select(setdiff(tskBat$feature_names, "fk_NO3")) ) %>% as.data.table()
  
  flt("importance", learner = as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity') ))$calculate(tsk_fk_RedD) %>% as.data.table()
  
  
  
  # 6.1.2  feature importance filters
  as.data.table(mlr_learners)[ sapply(properties, function(x) "importance" %in% x)]
  lrn("classif.ranger")$param_set$levels$importance
  
  
  # 6.2 wrapper methods
  
  library(mlr3fselect)
  
  instance <- fselect(
    fselector = fs("sequential"),
    task =  tsk_fk_BatD$clone(deep = T),
    learner = as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity')),
    resampling = rsmp("cv", folds = 3),
    measure = msr("classif.ce")
  ) # does rsmp cv - takes a bit, but works
  instance$archive %>% as.data.table %>% filter(batch_nr == 1) %>% dplyr::select(1:15)
  instance$archive %>% as.data.table %>% filter(batch_nr == 1) %>% dplyr::select(features, n_features)
  
  autoplot(instance, type = "performance")
  
  e_instance = fsi(
    task = tsk_fk_BatD$clone(deep = T),
    learner = as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity')),
    resampling = rsmp("cv", folds = 6),
    measures = mlr3::msr("classif.acc"),
    terminator = trm("none"))
  
  tmp <- fselect(
    fselector = fs("rfe", n_features = 2, feature_fraction = .9, aggregation = 'rank'), 
    task = tsk_fk_BatD$clone(deep = T),
    learner = as_learner(po("imputemedian") %>>% lrn('classif.ranger', importance = 'impurity')),
    resampling = rsmp("cv", folds = 6),
    measure = msr('classif.acc'),
    terminator = trm('none'))
  
  tmp$result_y # 0.7679055 
  tmp$result_feature_set #
  
  
  
} # mlr3book examples

if(F){
  
} # fk examples



  if(F){
  optimizer = mlr3fselect::fs("rfe",
                              n_features = 2,
                              #feature_number = 1,
                              feature_fraction = 0.9,
                              aggregation = "rank")
  
  e_instance = fsi(
    task = tsk_Batv_all,
    learner = learner0,
    resampling = rsmp("cv", folds = 6),
    measures = mlr3::msr("classif.acc"),
    terminator = trm("none"))
  
  optimizer$optimize(e_instance)
  e_instance$result
  e_instance$result_y # .7880372 
  e_instance$result_feature_set
  
  # We fit a final model with the optimized feature set to make predictions on new data.
  task$select(e_instance$result_feature_set)
  
  tmp <- fselect(
    fselector = fs("rfe", n_features = 2, feature_fraction = .9, aggregation = 'rank') , # 28 /.812 
    # fselector = fs("rfe", n_features = 2,  feature_number = 1, aggregation = 'rank') , # 25 /.7995548
    task = tsk_Batv_all,
    learner = learner0,
    resampling = rsmp("cv", folds = 6),
    measure = msr('classif.acc'),
    terminator = trm('none'))
  
  tmp$result_y # 0.7679055 
  tmp$result_feature_set # 115 features # 53/.787 # 28/.812 # 25 /.7995548 (w feature_number = 1)
  
  tmp <- fselect(
    fselector = fs("genetic_search") , # sbs 28 /.812 
    task = tsk_Batv_all,
    learner = learner0,
    resampling = rsmp("cv", folds = 6),
    measure = msr('classif.acc'),
    terminator = trm('none'))
  
  tmp$result_y # 0.7679055 
  tmp$result_feature_set #
  
  if(F){ # auto selector
    af0 = auto_fselector(
      fselector = fs("rfe", n_features = 2, feature_fraction = .9, aggregation = 'rank'),
      learner = lrn("classif.xgboost", eta = .1),
      resampling = rsmp("cv", folds = 6),
      measure = msr("classif.acc"),
      terminator = trm("none")
    )
    af0
    af1 = auto_fselector(
      fselector = fs("rfe", n_features = 2, feature_fraction = .9, aggregation = 'rank'),
      learner = learner0,
      resampling = rsmp("cv", folds = 6),
      measure = msr("classif.acc"),
      terminator = trm("none")
    )
    
    tsks <- list(tsk_Bat_all, tsk_Batv_all)
    (design = benchmark_grid(tsks, list(af0, af1), rsmp("cv", folds = 3)))
    bmr = benchmark(design)
    bmr$aggregate(msr("classif.acc"))
    as.data.table(bmr)[, .(learner_id, classif.acc)]
    
  } # auto selector
  
  if(F){ # big selector
    
    lrn_featureless <- lrn('classif.featureless', method = 'weighted.sample')
    Bat_all_fless <- resample(tsk_Bat_all, lrn_featureless, cv3)
    Bat_all_fless$aggregate(measure) #  0.8258
    
    
    Bat_all_features <- fselect(
      fselector = fs("rfe", n_features = 2, feature_fraction = .9, aggregation = 'rank') , # 19 /0.8826649 
      # fselector = fs("rfe", n_features = 2,  feature_number = 1, aggregation = 'rank') , # 25 /.7995548
      task = tsk_Bat_all,
      learner =  lrn("classif.xgboost", eta = .1),
      resampling = rsmp("cv", folds = 8),
      measure = msr('classif.acc'),
      terminator = trm('none'))
    
    Bat_all_features$result_y # 9/ 0.882
    Bat_all_features$result_feature_set # 1
    
    # verdict - iga jookusga saab eri muutujad, eriti MEM on labiilne. FK on stabiilsem: usual suspects: biokeemiline_hapnikutarve_bht5" "elektrijuhtivus_proovivotul"  "keemiline_hapnikutarve_dikromaatne" "nitraatlammastik_no3n"  
  }# big selector
  
  
  # visualise feature elimination
  if(F){
    library(viridisLite)
    library(mlr3misc)
    
    data = as.data.table(Bat_all_features$archive)
    data[, n:= map_int(importance, length)]
    
    ggplot(data, aes(x = n, y = classif.acc)) +
      geom_line(
        color = viridis(1, begin = 0.5),
        linewidth = 1) +
      geom_point(
        fill = viridis(1, begin = 0.5),
        shape = 21,
        size = 3,
        stroke = 0.5,
        alpha = 0.8) +
      xlab("Number of Features") +
      scale_x_reverse() +
      theme_minimal()
  } # ggplot e_instance$result
  
  
} # feature selection/elimination




