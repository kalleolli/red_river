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

# discover working directory; expects red_river.Rproj  in the  ~/Documents hierarchy
reddir <- list.files(path = '~/Documents', full.names = TRUE, recursive = TRUE, pattern = 'red_river.Rproj') %>% dirname()
  
load(paste0(reddir, '/dat/red_river_1.rda')) # load tsks [list of 12]; glrn_xx_lst [list of 6 graph learners], which were created in red_river_1.R
# names(tsks)
# names(glrn_xx_lst)

# benchmark ####

# tuned hyperparameter collections of 50 with the respective tasks has benchmarks

# make naive featureless, xg and rg learners for benchmarking
if(T){
  # ranger (random forest) does not accept missing data, we impute median values with graph learner
  glrn_fl <- as_learner(po("imputemedian") %>>% lrn("classif.featureless", method = 'weighted.sample', predict_type = 'prob'))
  glrn_xg <- as_learner(po("imputemedian") %>>% lrn("classif.xgboost", predict_type = 'prob'))
  glrn_rg <- as_learner(po("imputemedian") %>>% lrn("classif.ranger", predict_type = 'prob', importance = 'impurity'))
  
  glrn_xg$id <- 'glrn_0_xg_0'
  glrn_rg$id <- 'glrn_0_rg_0'
  glrn_fl$id <- 'glrn_0_fl_0'
  
  untuned_learners <- list(
    fl = glrn_fl,
    xg = glrn_xg,
    rg = glrn_rg
)
} # provides untuned_learners: rg, xg, and fl (featureless)

message("Start making a list of learners with hyperband tuned parameters included  \n\n ")

if(T){
  # make a list of learners with hyperband tuned parameters included  
  # list of 12 - each is combo of Red/Bat, x 6 feature groups (cmb, fk, ap, pk, mk, st)
  # each of 12 is 50 header x 6 learners = 300 learners per list slot
  # loop over 12 hyp_files; load (as hyptune_heads)
  
  # list of 12 files, which store the tuned hyperparameter sets
  hyp_files <- list.files(paste0(reddir,'/dat/')) %>% grep('hyptune_tsk_', .,  value = T) # 12
  
  hyp_lst <- list() # final list of 12, each with 300 learners
  for(hf in 1:length(hyp_files)){
    load(paste0(reddir,'/dat/', hyp_files[hf])) # loads hyptune_heads; list of length 6; 
    # each is a data.table with 50 best models hyperparameter sets for that learner
    # the 2nd column of the data table - x_domain - contains the hyperparameter sets
    tmp <- lapply(seq_along(hyptune_heads), function(i){
      learner_name <- names(hyptune_heads)[i]
      if(grepl('_rg', learner_name)) learner1 <- glrn_rg$clone(deep = T) else learner1 <- glrn_xg$clone(deep = T)
      lst <- lapply(1:nrow(hyptune_heads[[i]]), function(x){learner2 <- learner1$clone(deep = T); learner2$param_set$set_values(.values = hyptune_heads[[i]]$x_domain[[x]]);  learner2$id <- paste(learner_name, x, sep = '_'); return(learner2)})
      return(lst)
    })
    hyp_lst[[hf]] <- unlist(tmp)
    message('Processing hypfile ', hf, '\n' )
  }
  names(hyp_lst) <- sub('hyptune_tsk_','', hyp_files) %>% sub('.rda','',.)
  
} # provides hyp_lst - hyperparameter loaded learners, ready to be benchmarked

message("Start making a list of of benchmark grids  \n ")



if(T){
 
## tuned model benchmark grid ####
  bm_grid_tuned <- list()
  for(i in 1:length(hyp_lst)){
    bm_grid_tuned[[i]] <- mlr3::benchmark_grid(
      tasks = tsks[i],
      learners = c(hyp_lst[[i]]), #  
      rsmp('cv', folds = 5))
    message(' Processing grd_lst ', i, '\n')
  }
 
## naive model benchmark grid ####
  bm_grid_naive <- mlr3::benchmark_grid(
    tasks = tsks, # 12 tasks cmb + 5 feature groups
    learners = untuned_learners, # featureless, ranger, xgboost graph learners
    rsmp('cv', folds = 5))
  
save(bm_grid_tuned, bm_grid_naive, file = paste0(reddir, '/dat/bm_grids.rda'))
  
} # provides and saves 3636  x  3 ./dat/bm_grids.rda - benchmark grid object

message("Benchmark grid saved in ./dat/bm_grids.rda  \n\n ")
message(" proceed with red_river_4.R \n")

