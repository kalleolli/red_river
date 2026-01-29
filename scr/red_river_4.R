
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
library(tidyr)


reddir <- list.files(path = '~/Documents', full.names = TRUE, recursive = TRUE, pattern = 'red_river.Rproj') %>% dirname()

load(paste0(reddir, '/dat/bm_grids.rda')) # loads bm_grid_tuned and bm_grid_naive
 
message("Tuned benchmarking \n\n ")
  
system.time(bmr_tuned <- parallel::mclapply(bm_grid_tuned, mlr3::benchmark, mc.cores = 12)) # takes time

message("Naive benchmarking \n\n ")

# make 50 replicates of naive benchmark grid to match the 50 tuned models
system.time(bmr_naive <- parallel::mclapply(1:50, function(x){res <- benchmark(bm_grid_naive); res}, mc.cores = 10)) # takes time



bmr_tuned_ag <- lapply(bmr_tuned, function(x){x$aggregate(msr("classif.ce"))[, .(task_id, learner_id, classif.ce)]}) %>% bind_rows(.id = 'batch_id') # 3636 x 4
bmr_naive_ag <- lapply(bmr_naive, function(x){x$aggregate(msr("classif.ce"))[, .(task_id, learner_id, classif.ce)]}) %>% bind_rows(.id = 'batch_id') # 1800 x 4

## combine tuned and naive benchmark results ####
bmr_all <- rbind(bmr_tuned_ag, bmr_naive_ag) # 5400 x 4

## parse benchmarking categories ####

bmr_all <- strsplit(bmr_all$task_id, '_') %>% do.call(rbind, .) %>% as.data.table() %>% transmute(., red_id = sub('D','',V1), feat_id = V2) %>% bind_cols(bmr_all, .);
# table(bmr_a$feat_id) # all 600 combinations of feature groups present
bmr_all <- strsplit(bmr_all$learner_id, '_') %>% do.call(rbind, .) %>% as.data.table() %>% transmute(., hyp_id = V2, mod_id = sub('b','',V3), nr = V4) %>% bind_cols(bmr_all, .);

# 5400 x 9

if(F){
  # bmr_all summary
  table(bmr_all$hyp_id) # 0 - naive , 1, 2, D
  table(bmr_all$mod_id) # rg xg fl
  table(bmr_all$red_id) # Bat Red
  table(bmr_all$feat_id) # ap cmb  fk  mk  pk  st 
} # bmr_all stats

  


message("Save benchmarking \n\n ")
save(bmr_all, file = paste0(reddir, '/dat/bmr_all.rda'))

SuppTable2 <- bmr_all %>% mutate(hyp_id = ifelse(hyp_id == '0', 'Naive', 'Tuned')) %>% 
  summarise(., mean_ce = signif(median(classif.ce),3), .by = c(feat_id, red_id, mod_id, hyp_id)) %>% 
  tidyr::pivot_wider(., id_cols=c(feat_id, hyp_id), names_from = c(red_id, mod_id), values_from = c(mean_ce)) %>% 
  arrange(feat_id, hyp_id)

write.table(SuppTable2, file = paste0(reddir,'/tables/red_river_SuppTable21.csv'), col.names = TRUE, sep = '\t' , quote = FALSE, row.names = FALSE)


message("Benchmark grid saved in ./dat/bmr.rda  \n\n ")
message("proceed with importance calculations -- red_river_5.R \n")


if(F){
  # xg vs rg tuned comparison table
  
  b <- filter(SuppTable2, hyp_id == 'Tuned') 
  t.test(x=b$Red_xg, y=b$Red_rg, paired = TRUE) # p-value = 0.3632; tuned Red not different
  t.test(x=b$Bat_xg, y=b$Bat_rg, paired = TRUE) # p-value = 0.6446; tuned Bat not different
  
  # xg vs rg untuned comparison table
  c <- filter(SuppTable2, hyp_id == 'Naive') 
  t.test(x=c$Red_xg, y=c$Red_rg, paired = TRUE) # p-value = 0.04657; naive Red marginally different  
  t.test(x=c$Bat_xg, y=c$Bat_rg, paired = TRUE) # p-value = 0.05988; naive Bat not different  
  t.test(x=c$Bat_fl, y=c$Bat_rg, paired = TRUE) # all combos significantly different
  
  # xg and rg tuned vs untuned comparison table
  d <- select(SuppTable2, hyp_id, Bat_xg, feat_id) %>% pivot_wider(id_cols = feat_id, names_from = hyp_id, values_from = Bat_xg) 
  t.test(x=d$Naive, y=d$Tuned, paired = TRUE) # p-value = 0.1813 Bat_xg not different :(
  d <- select(SuppTable2, hyp_id, Bat_rg, feat_id) %>% pivot_wider(id_cols = feat_id, names_from = hyp_id, values_from = Bat_rg) 
  t.test(x=d$Naive, y=d$Tuned, paired = TRUE) # p-value = 0.2978 Bat_rg not different 
  
  d <- select(SuppTable2, hyp_id, Red_xg, feat_id) %>% pivot_wider(id_cols = feat_id, names_from = hyp_id, values_from = Red_xg) 
  t.test(x=d$Naive, y=d$Tuned, paired = TRUE) #  p-value = 0.02789 Red_xg different :(
  d <- select(SuppTable2, hyp_id, Red_rg, feat_id) %>% pivot_wider(id_cols = feat_id, names_from = hyp_id, values_from = Red_rg) 
  t.test(x=d$Naive, y=d$Tuned, paired = TRUE) # p-value = 0.3833 Red_rg not different 
  
  c(d$Naive-d$Tuned) %>% t.test()
  
  
  # benchmark verdict: with un-tuned models, rg is much better than xg. However, rg does not improve with tuning, while xg does and finally beats rg on classif.ce front.
  
  # combo benchmark verdict:
  # untuned rg is better on BatD adbd BatW, xgb is better on red Family
  # sweet spot seems xgboost on D tuned learner
  # when tuned, rg nad xgb are the same, neither is there any difference if the tuning is D, 1 or 2. Parsimony - pick D
  
  # benchmark verdicts:
  # 1. fk is most important - whereever it is in (fk, cmb, combo) predictive power is good. 
  # 2. Bat is good with every feature set. Red makes clear difference - fk makes the day, others much worse. 
  # 3. For Bat cmb is best, beats fk alone
  # 4. For Red fk alone beats cmb and combo
  # 5. ap is always the worst.
  # 6. No big difference between the 1, 2, D models, but choose one by the end of the day.
  
  
  # https://mlr3book.mlr-org.com/chapters/chapter11/large-scale_benchmarking.html
}


