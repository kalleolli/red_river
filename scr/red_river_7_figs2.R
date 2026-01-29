
# providers feature importance figs 5 and 6

library(tidyverse)
library(vegan) # decostand to scale feature impact
library(data.table)
library(cowplot) # plot_grid


reddir <- list.files(path = '~/Documents', full.names = TRUE, recursive = TRUE, pattern = 'red_river.Rproj') %>% dirname()

# load objects
load(file = paste0(reddir, '/dat0/super_task.rda'))

load(file = paste0(reddir, '/dat/all_imp_mcl_v2.rda')) #  loads all_imp_mcl 582 Mb object; accumulated local effects; created in red_river_5.R
load(file = paste0(reddir, '/dat/bmr_all.rda')) # loads bmr_all in red_river_4.R

bmr_tuned_ag <- filter(bmr_all, hyp_id != '0')#  %>% distinct(batch_id, .keep_all = T) # 1800 x 10
# TODO: describe  objects

# shrink down all_imp_mcl to save space
iml.imp <- lapply(all_imp_mcl, '[[', 1) # 12 Mb
rm(all_imp_mcl)

# feature importance boxplots ####
if(T){
  
# scale iml importance estimates with decostand, max and total
  
iml.importance <- lapply(seq_along(iml.imp), function(i){data.table(iml.imp[[i]])[, mean(importance), by = feature ] %>% 
      transmute(feature = feature, importance = V1, impmax = as.numeric(decostand(V1, method = 'max')), imprange = as.numeric(vegan::decostand(V1, method = 'range'))) %>%  
    bind_cols(., bmr_tuned_ag[i, ]) }) %>% bind_rows() # 1.7 Mb object; 22800 x 9

#### FIG 5 cmb  ####
# # boxplots of feature group importance from cmb models

iml.dat <- filter(iml.importance, feat_id == 'cmb', hyp_id != '0') %>% 
  mutate(red_id = factor(red_id, levels = c('Bat','Red'), labels = c('Batrachospermum', 'All red algae')))
  
  Fig5 <- ggplot(iml.dat) + geom_boxplot(aes(y = imprange, x = feature), outliers = FALSE) + 
    facet_grid(cols = vars(red_id), scales = 'free_y')  +  
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    theme(legend.position="none") + 
    labs(x = 'Variable group', y = 'Scaled importance') + # or Feature group
    scale_x_discrete(labels=c("ap" = "Bedrock", "fk" = "Hydro Chem", "mk" = "Land use", 'pk' = 'Land cover','st' = 'Substrate'))
  
  ggsave(Fig5, filename = paste0(reddir,'/figs/red_river_fig5.pdf'),  width = 6, height = 6)
  
  #### FIG 6 fk  ####
  # zoom in to hydro-chemical variables only
  
  iml.dat <- filter(iml.importance, feat_id == 'fk') %>% 
    mutate(red_id = factor(red_id, levels = c('Bat','Red'), labels = c('Batrachospermum', 'All red algae')), feature = sub('fk_','',feature) %>% 
             sub('_',' ',.) )
  
  Fig6 <- ggplot(iml.dat) + geom_boxplot(aes(y = imprange, x = feature), outliers = FALSE) + 
    facet_grid(cols = vars(red_id), scales = 'free_y')  +  
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none") + 
    labs(x = 'Variable', y = 'Scaled importance')
  
  ggsave(Fig6, filename = paste0(reddir, '/figs/red_river_fig6.pdf'),  width = 6, height = 6)
  
} # provides figs/red_river_fig5.pdf (cmb imp), figs/red_river_fig6.pdf (fk imp)



# Supplementary table 1 of mean feature importance ####
if(F){
  # table(iml.importance$hyp_id)
  
  tmp <- filter(iml.importance, hyp_id != '0') %>% summarise(mean_imp = signif(mean(importance),3), .by = c(feature, red_id, mod_id))
  
  # feature group importance table
  filter(tmp, !grepl('_', feature)) %>% 
    pivot_wider(id_cols=feature, names_from = c(red_id, mod_id), values_from = c(mean_imp)) %>% 
    write.table(., file = paste0(reddir,'/tables/red_river_SuppTable_a.csv'), col.names = TRUE, sep = '\t' , quote = FALSE, row.names = FALSE) 
  
  # individual feature importance table
  filter(tmp, grepl('_', feature)) %>% 
    pivot_wider(id_cols=feature, names_from = c(red_id, mod_id), values_from = c(mean_imp)) %>% 
    write.table(., file = paste0(reddir,'/tables/red_river_SuppTable_b.csv'), col.names = TRUE, sep = '\t' , quote = FALSE, row.names = FALSE) 
  
  
  
}

# Supplementary tables of benchmarking ####
if(F){
  reddir <- list.files(path = '~/Documents', full.names = TRUE, recursive = TRUE, pattern = 'red_river.Rproj') %>% dirname()
  load(file = paste0(reddir, '/dat/bmr2.rda')) # loads bmr, bmr_a; created in red_river_4.R
  
  
  
  tmp_bmr <- summarise(bmr_all, mean_ce = signif(mean(classif.ce),3), .by = c(feat_id, red_id, mod_id, hyp_id))
  pivot_wider(tmp_bmr, id_cols=c(feat_id, hyp_id), names_from = c(red_id, mod_id), values_from = c(mean_ce)) %>% print(n=Inf)
  
  a <- bmr_all %>% mutate(hyp_id = ifelse(hyp_id == '0', 'Naive', 'Tuned')) %>% 
    summarise(., mean_ce = signif(median(classif.ce),3), .by = c(feat_id, red_id, mod_id, hyp_id)) %>% 
    pivot_wider(., id_cols=c(feat_id, hyp_id), names_from = c(red_id, mod_id), values_from = c(mean_ce))
  
  
  
  a0 <- tmp_bmr %>% mutate(hyp_id = ifelse(hyp_id == '0', 'Naive', 'Tuned')) %>% 
    summarise(., mean_ce = signif(mean(mean_ce),3), .by = c(feat_id, red_id, mod_id, hyp_id)) %>% 
    pivot_wider(., id_cols=c(feat_id, hyp_id), names_from = c(red_id, mod_id), values_from = c(mean_ce))
  
  
  
  
  
  
  load(paste0(reddir, '/dat/red_river_1.rda')) # load tsks
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
  } # provides untuned_learners
  
  
  bm_grid_naive <- mlr3::benchmark_grid(
    tasks = tsks, # 12 tasks cmb + 5 feature groups
    learners = untuned_learners, # featureless, ranger, xgboost graph learners
    rsmp('cv', folds = 5))
  
  system.time(naive_list <- parallel::mclapply(1:50, function(x){res <- benchmark(bm_grid_naive); res}, mc.cores = 10)) # 2847s
  
  bmr_a <- lapply(bmr, function(x){x$aggregate()}) %>% bind_rows()
  tmp <- lapply(naive_list, function(x){x$aggregate()}) %>% bind_rows()
  
  # aggregate naive results
  bmr_tunesd <- lapply(bmr, function(x){x$aggregate(msr("classif.ce"))[, .(task_id, learner_id, classif.ce)]}) %>% bind_rows(.id = 'batch_id') # 3636 x 4
  bmr_naive <- lapply(naive_list, function(x){x$aggregate(msr("classif.ce"))[, .(task_id, learner_id, classif.ce)]}) %>% bind_rows(.id = 'batch_id') # 1800 x 4
  bmr_all <- rbind(bmr_tunesd, bmr_naive) # 5436    4
  
  # parse bmr categories; delete capital D
  bmr_all <- strsplit(bmr_all$task_id, '_') %>% do.call(rbind, .) %>% as.data.table() %>% transmute(., red_id = sub('D','',V1), feat_id = V2) %>% bind_cols(bmr_all, .);
  
 
  # table(bmr_a$feat_id) # all 600 combinations of feature groups present
  bmr_all <- strsplit(bmr_all$learner_id, '_') %>% do.call(rbind, .) %>% as.data.table() %>% transmute(., hyp_id = V2, mod_id = sub('b','',V3), nr = V4) %>% bind_cols(bmr_all, .);
  
  data.table::rbindlist(bmr_a, bmr_naive)
  do.call(rbind, list(bmr_a, bmr_naive), fill = TRUE)

    # names(hyp_lst) # from red_river_3.R; 12 learner lists, 300 hyperparameter tuned learners each, 50 top ones with 1, 2, D models x rg, xg
  # names(tsks)    # from red_river_1.R
  
  bm_grid_tuned_ap_Bat <- mlr3::benchmark_grid(
    tasks = tsks['tsk_ap_BatD'], 
    learners = hyp_lst[['ap_BatD']],
    rsmp('cv', folds = 10))
  system.time(bmr_tuned_ap_Bat <- benchmark(bm_grid_tuned_ap_Bat)) # 2507 sek
  bmr_tuned_ap_Bat$aggregate(msr("classif.ce"))[, .(task_id, learner_id, classif.ce)]  # OK
  
} # benchmark suppl table is in script 4



  # benchmark, and record both loss functions - training and test sets - if difference is large, we are dealing with overfitting
  if(F){
    # ToRedRiv_ms.R, line 31:
    # mlr3 book p. 52 - As a rule of thumb, it is common to use 2/3 of the data for training and 1/3 for testing as this provides a reasonable trade-off between bias and variance of the generalization performance estimate (Kohavi 1995; Dobbin and Simon 2011). Justifying tuning with 3x cross-validation
    
    # pinguin book example
    if(F){ # pinguin book example
      # Define task, learner, and resampling strategy
      tsk_penguins = tsk("penguins")
      lrn_rpart = lrn("classif.rpart")
      rsmp_cv = rsmp("cv", folds = 3)
      
      # Perform resampling
      rr = resample(tsk_penguins, lrn_rpart, rsmp_cv)
      
      acc = rr$score(msr("classif.ce"))
      acc[, .(iteration, classif.ce)]
      
      rr$aggregate(msr("classif.ce"))
      rr$aggregate(msr("classif.acc"))
      
      
      # Initialize a vector to store training scores
      train_scores = numeric(rr$iters)
      
      # Loop through each resampling iteration
      for (i in seq_len(rr$iters)) {
        train_set = rr$resampling$train_set(i)
        lrn_rpart$train(tsk_penguins, row_ids = train_set)
        prediction = lrn_rpart$predict(tsk_penguins, row_ids = train_set)
        train_scores[i] = prediction$score(msr("classif.ce"))
      }
      
      # Print training scores
      print(train_scores) # training scores [0.04366812 0.04803493 0.05217391] are less than test scores [0.06956522 0.07826087 0.04385965]
      acc$classif.ce
      
      train_scores/acc$classif.ce  # 0.6285714 0.6142857 1.1882353}
    } # pinguin book example

    # Naive red
    if(F){
      tsks # 12 choose "tsk_fk_BatD"  "tsk_fk_RedD" 
      # glrn_rg glrn_xg # naive learners
      rsmp_cv = rsmp("cv", folds = 3)
      rrg <- resample(tsks[['tsk_fk_RedD']], glrn_rg, rsmp_cv)
      rrx <- resample(tsks[['tsk_fk_RedD']], glrn_xg, rsmp_cv)
      
      rrg$score(msr("classif.ce"))$classif.ce
      rrx$score(msr("classif.ce"))$classif.ce
      
      train_scores_rg <- train_scores_rx <- numeric(rrg$iters)
      for (i in seq_len(rrg$iters)) {
        train_set = rrg$resampling$train_set(i)
        glrn_rg$train(tsks[['tsk_fk_RedD']], row_ids = train_set)
        prediction = glrn_rg$predict(tsks[['tsk_fk_RedD']], row_ids = train_set)
        train_scores_rg[i] = prediction$score(msr("classif.ce"))
        
        train_set = rrx$resampling$train_set(i)
        glrn_xg$train(tsks[['tsk_fk_RedD']], row_ids = train_set)
        prediction = glrn_xg$predict(tsks[['tsk_fk_RedD']], row_ids = train_set)
        train_scores_rx[i] = prediction$score(msr("classif.ce"))
      }  
      print(train_scores_rg)  
      print(train_scores_rx)  
      
      mean(train_scores_rg/rrg$score(msr("classif.ce"))$classif.ce) # 0.3473085
      mean(train_scores_rx/rrx$score(msr("classif.ce"))$classif.ce) # 0.6576643
    } # Naive red
    
    # Hyptuned red
    train_scores_fk <- test_scores_fk <- matrix(0, nrow = 300, ncol = 3)
    if(T){
      rsmp_cv = rsmp("cv", folds = 3)
      hyp_lrnrs <- hyp_lst[['fk_BatD']] # length 300 # length(hyp_lst) = 12
      
      for(i in 1:length(hyp_lrnrs)){
        glrn_tmp <- hyp_lrnrs[[i]]
        rr <- resample(tsks[['tsk_fk_RedD']], glrn_tmp, rsmp_cv)
        test_scores <- rr$score(msr("classif.ce"))$classif.ce
        
        train_scores <- numeric(rr$iters)
        for (j in seq_len(rr$iters)) {
          train_set = rr$resampling$train_set(j)
          glrn_tmp$train(tsks[['tsk_fk_RedD']], row_ids = train_set)
          prediction = glrn_tmp$predict(tsks[['tsk_fk_RedD']], row_ids = train_set)
          train_scores[j] = prediction$score(msr("classif.ce"))
        }
        train_scores_fk[i,] <- train_scores
        test_scores_fk[i,] <- test_scores
        print(i)
      }
      
      train_scores_fk/test_scores_fk
      
      
      
    }
}
  
  