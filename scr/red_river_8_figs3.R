
# providers ALE plot fig 7

library(tidyverse)
library(data.table)
library(ggside) # to add data density distribution
library(cowplot) # plot_grid

# assumes red_river.Rproj  in the  ~/Documents hierarchy
reddir <- system("find ~/Documents -name 'red_river.Rproj' ", intern = T) %>% sub('red_river.Rproj','', .)

# load objects
load(file = paste0(reddir, 'dat0/super_task.rda'))

load(file = paste0(reddir, 'dat/all_imp_mcl_v2.rda')) #  loads all_imp_mcl 582 Mb object; accumulated local effects
load(file = paste0(reddir, 'dat/bmr2.rda')) # loads bmr, bmr_a

# shrink down all_imp_mcl to save space
ale.imp <- lapply(all_imp_mcl, '[[', 2) # 552 Mb
rm(all_imp_mcl)


ale.df <- lapply(seq_along(ale.imp), function(i){bind_cols(ale.imp[[i]], bmr_a[i,red_id:batch_id])} ) %>% bind_rows() %>% filter(batch_id != '0') %>% mutate(hyp_id = factor(hyp_id, levels = c('D','1','2'), labels = c('Default','rbv1','rbv2') )) # 1494000       9

# select
idx <- which(bmr_a$feat_id == 'fk' & bmr_a$red_id == 'Red' & bmr_a$mod_id == 'rg' & bmr_a$batch_id != '0') # length 150 = 3 mod_id x 50 hyp-headers


feats <- c('fk_BOD', 'fk_Depth','fk_TN','fk_O2p') # best features according to permutational feature importance
featnames <- c('Biological oxygen demand', 'Depth','Total nitrogen','Oxygen saturation')
ale_fig_base <- filter(ale.df, feature %in% feats, red_id == 'Red' , mod_id == 'rg', feat_id == 'fk')


ale_fig_lst <- X <-  list()
for(i in 1:length(feats)){
  X[[i]] <- filter(ale_fig_base, feature == feats[i])$X %>% unique()
  rug_base <- data.table(X = unlist(dplyr::select(super_task, feats[i])), y = 0)
  ale_fig_lst[[i]] <- ggplot(filter(ale_fig_base, feature == feats[i]), aes(x = X, y = value, col = hyp_id)) +
    geom_line(aes(group = paste0(batch_id, hyp_id)), alpha = .05) +
    theme(legend.position="none") +
    geom_rug(mapping = aes(x = X, y = y), data = rug_base, sides = 'b', inherit.aes = F, col = rgb(.5,0,0, alpha = .1), length = unit(0.04, "npc")) +
    geom_xsidedensity(data = rug_base, mapping = aes(y = after_stat(density)), position = "stack", outline.type = 'upper',  col = 'black') + 
    ggside(x.pos = "bottom") +
    #theme(axis.text.x = element_text(angle = 90, vjust = .5))
    theme(ggside.axis.text.y = element_blank()) +
    labs(title = featnames[i], x = 'Feature', y = 'Relative importance')
}

# add legend to last panel
ale_fig_lst[[i]] <- ale_fig_lst[[i]] + 
  theme(legend.position = c(0.8, 0.4), legend.background = element_rect(fill = NA), legend.key = element_rect(fill = NA, color = NA)) + 
  guides(colour = guide_legend(override.aes = list(linewidth = 1, alpha = 1))) +
  labs(color = 'Tuning spaces')

plot_grid(plotlist = lapply(seq_along(ale_fig_lst), function(i){ale_fig_lst[[i]] + xlim(quantile(X[[i]],  probs=c(0.05, .95)))}), ncol = 2) %>% 
  ggsave(filename = paste0(reddir,'figs/red_river_fig7.pdf'), width = 8, height = 8)

