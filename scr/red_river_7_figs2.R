
# providers feature importance figs 5 and 6

library(tidyverse)
library(vegan) # decostand to scale feature impact
library(data.table)
library(cowplot) # plot_grid


# no prior calculations, takes data from dat0 folder

# discover working directory
# assumes red_river.Rproj  in the  ~/Documents hierarchy
reddir <- system("find ~/Documents -name 'red_river.Rproj' ", intern = T) %>% sub('red_river.Rproj','', .)

# load objects
load(file = paste0(reddir, 'dat0/super_task.rda'))

load(file = paste0(reddir, 'dat/all_imp_mcl_v2.rda')) #  loads all_imp_mcl 582 Mb object; accumulated local effects
load(file = paste0(reddir, 'dat/bmr2.rda')) # loads bmr, bmr_a

# TODO: describe  objects

# shrink down all_imp_mcl to save space
iml.imp <- lapply(all_imp_mcl, '[[', 1) # 12 Mb
rm(all_imp_mcl)

# feature impact boxplots ####

if(T){
  
  # scale iml importance estimates with decostand, max and total
  
  iml.importance <- lapply(seq_along(iml.imp), function(i){data.table(iml.imp[[i]])[, mean(importance), by = feature ] %>% 
      transmute(feature = feature, importance = V1, impmax = as.numeric(decostand(V1, method = 'max')), imprange = as.numeric(decostand(V1, method = 'range'))) %>%  bind_cols(., bmr_a[i, red_id:batch_id]) }) %>% bind_rows() # 1.7 Mb object; 24764 x 9
  
  
  #### FIG 5 cmb  ####
  
  iml.dat <- filter(iml.importance, feat_id == 'cmb', hyp_id != '0') %>% mutate(red_id = factor(red_id, levels = c('Bat','Red'), labels = c('Batrachospermum', 'All red algae')))
  
  Fig5 <- ggplot(iml.dat) + geom_boxplot(aes(y = imprange, x = feature), outliers = FALSE) + 
    facet_grid(cols = vars(red_id), scales = 'free_y')  +  
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none") + 
    labs(x = 'Feature group', y = 'Scaled importance') +
    scale_x_discrete(labels=c("ap" = "Bedrock", "fk" = "Hydro Chem", "mk" = "Land use", 'pk' = 'Land cover','st' = 'Substrate'))
  
  ggsave(Fig5, filename = paste0(reddir,'figs/red_river_fig5.pdf'),  width = 6, height = 6)
  
  ggplot(iml.dat) + geom_boxplot(aes(y = importance, x = feature), outliers = FALSE) + 
    facet_grid(rows = vars(red_id), scales = 'free_y')  +  
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none") + 
    labs(x = 'Feature group', y = 'Importance') +
    scale_x_discrete(labels=c("ap" = "Bedrock", "fk" = "Hydro Chem", "mk" = "Land use", 'pk' = 'Land cover','st' = 'Substrate'))
  
  #### FIG 6 fk  ####
  
  iml.dat <- filter(iml.importance, feat_id == 'fk') %>% mutate(red_id = factor(red_id, levels = c('Bat','Red'), labels = c('Batrachospermum', 'All red algae')), feature = sub('fk_','',feature) %>% sub('_',' ',.) )
  
  Fig6 <- ggplot(iml.dat) + geom_boxplot(aes(y = imprange, x = feature), outliers = FALSE) + 
    facet_grid(cols = vars(red_id), scales = 'free_y')  +  
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none") + 
    labs(x = 'Feature', y = 'Scaled importance')
  
  ggsave(Fig6, filename = paste0(reddir, 'figs/red_river_fig6.pdf'),  width = 6, height = 6)
  
} # provides figs/red_river_fig5.pdf (cmb imp), figs/red_river_fig6.pdf (fk imp)








