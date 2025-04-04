
# red_riv figs

library(tidyverse)
library(vegan) # decostand to scale feature impact
library(data.table)
library(ggside) # to add data density distribution
library(cowplot) # plot_grid
library(ggspatial) # scales and N arrows
library(rnaturalearth)
library(rnaturalearthdata)
library(sf)
library(RColorBrewer)

# set working directory
reddir <- system("find ~/Documents -name 'red_river.Rproj' ", intern = T) %>% sub('red_river.Rproj','', .)

rm(list=grep('reddir',ls(), value = T, invert = T)) # garbage collect all but reddir

# load objects

load(file = paste0(reddir, 'dat/all_imp_mcl.rda')) #  loads all_imp_mcl 582 Mb object; accumulated local effects
load(file = paste0(reddir, 'dat/super_task.rda')) 
load(file = paste0(reddir, 'dat/bmr.rda')) # loads bmr, bmr_a

# TODO: describe  objects

# shrink down all_imp_mcl to save space
iml.imp <- lapply(all_imp_mcl, '[[', 2) # 12 Mb
ale.imp <- lapply(all_imp_mcl, '[[', 3) # 552 Mb
rm(all_imp_mcl)

# distribution maps ####

#super_task <- cbind(select(redCCM, date, veekogu_kkr, seirekoha_kkr), super_task)
#save(super_task, file = './dat/super_task.rda')
#load('/Users/olli/Documents/Projects/RedRiver/dat/red_river_ini.rda') # dat[305 x 49], red [1321 x 54], redh [1321 x 66], redCCM [1263   35]

load(paste0(reddir, './dat/est.rda')) # est 40 kB
load(paste0(reddir, './dat/red_rivers.rda')) # sampled rivers sf

if(T){

  basemap <- ggplot(data = est) +
    geom_sf(fill= "whitesmoke") + 
    geom_sf(data = red_rivers, aes(geometry = geometry), colour = 'gray', lwd = 0.1) + 
    geom_point(data = unique(select(super_task, X, Y)), aes(x=X, y=Y), color = 'darkgray', size = 1 ) + 
    annotation_scale(location = "bl", width_hint = 0.2) + xlab('Longitude') + ylab('Latitude') + 
    annotation_north_arrow(location = "bl", which_north = "true", pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"), style = north_arrow_fancy_orienteering) + theme(panel.background = element_rect(fill = "aliceblue"))
  
  
  # insert map for northern Europe
  neurope <- ne_countries( scale = "medium", returnclass = "sf") %>% st_crop(., xmin = 2, xmax = 30, ymin = 50, ymax = 70)
  
  ggm1 <- ggplot(data = neurope) + geom_sf(fill = 'whitesmoke')   + geom_sf(data = st_as_sfc(st_bbox(st_transform(est, 4326))), fill = NA, color = 'red', linewidth = .3 ) +
    ggspatial::coord_sf(xlim=c(2, 30), ylim = c(55, 70), expand = T) + theme_void()
  
  
  
  ## Figure 1 ####
  
  Figure1 <- cowplot::ggdraw() +
    draw_plot(basemap) +
    draw_plot(ggm1, scale = 0.2, halign = 0.08, valign = 0.82)
  
  ggsave2(Figure1, file = paste0(reddir, './figs/RedRiver_Figure1.pdf'), width = 7.5, height = 6.5)
  
  
  ## Figure 2 ####
  
  tmp <- select(super_task, X, Y, Audouniella.hermannii, Audouniella.chalybea) %>% pivot_longer(cols = c('Audouniella.hermannii', 'Audouniella.chalybea'), names_to = "Taxon", values_to = "count") %>% filter(count > 0)
  
  Figure2 <- ggplot(data = est) +
    geom_sf(fill= "whitesmoke") + 
    geom_sf(data = red_rivers, aes(geometry = geometry), colour = 'gray', lwd = 0.1) + 
    geom_point(data = tmp, aes(x = X, y = Y, color = Taxon)) +
    scale_color_manual(labels = c("Audouniella chalybea", "Audouniella hermannii"), values = brewer.pal(3, 'Dark2')) +
    theme(panel.background = element_rect(fill = "aliceblue"), 
          legend.title = element_blank(),
          legend.position = "inside", 
          legend.position.inside = c(0.15, 0.9), 
          # legend.box = element_blank(),
          legend.background = element_rect(fill = NA, colour = NA, inherit.blank = FALSE),
          legend.key = element_rect(fill = NULL, colour = NA)) + 
    xlab('Longitude') + ylab('Latitude')
  
  ggsave2(Figure2, file = paste0(reddir, './figs/RedRiver_Figure2.pdf'), width = 7.5, height = 6.5)
  
  
  ## Figure 3 ####
  tmp <- select(super_task, X, Y, Batrachospermum, Hildenbrandia.rivularis) %>% pivot_longer(cols = c("Batrachospermum", "Hildenbrandia.rivularis" ), names_to = "Taxon", values_to = "count") %>% filter(count > 0) %>% unique()
  
  Figure3 <- ggplot(data = est) +
    geom_sf(fill= "whitesmoke") + 
    geom_sf(data = red_rivers, aes(geometry = geometry), colour = 'gray', lwd = 0.1) + 
    geom_point(data = tmp, aes(x = X, y = Y, color = Taxon)) +
    scale_color_manual(labels = c("Batrachospermum", "Hildenbrandia rivularis"), values = brewer.pal(3, 'Dark2')) +
    theme(panel.background = element_rect(fill = "aliceblue"), 
          legend.title = element_blank(),
          legend.position = "inside", 
          legend.position.inside = c(0.15, 0.9), 
          # legend.box = element_blank(),
          legend.background = element_rect(fill = NA, colour = NA, inherit.blank = FALSE),
          legend.key = element_rect(fill = NULL, colour = NA)) + 
    xlab('Longitude') + ylab('Latitude')
  
  ggsave2(Figure3, file = paste0(reddir, './figs/RedRiver_Figure3.pdf'), width = 7.5, height = 6.5)
  
  ## Figure 4 ####
  
  tmp <- select(super_task, X, Y, Lemanea, Lemanea.fucina, Lemanea.mammilosa, Lemanea.fluviatilis, Lemanea.rigida) %>% pivot_longer(cols = c('Lemanea', 'Lemanea.fucina', 'Lemanea.mammilosa', 'Lemanea.fluviatilis', 'Lemanea.rigida' ), names_to = "Taxon", values_to = "count") %>% filter(count > 0) %>% unique() %>% arrange(., desc(Taxon))
  
  Figure4 <- ggplot(data = est) +
    geom_sf(fill= "whitesmoke") + 
    geom_sf(data = red_rivers, aes(geometry = geometry), colour = 'gray', lwd = 0.1) + 
    geom_point(data = tmp, aes(x = X, y = Y, color = Taxon)) +
    scale_color_manual(labels = c("Lemanea sp", "Lemanea fucina", 'Lemanea mammilosa', 'Lemanea fluviatilis', 'Lemanea rigida'), values = brewer.pal(5, 'Dark2')) +
    theme(panel.background = element_rect(fill = "aliceblue"), 
          legend.title = element_blank(),
          legend.position = "inside", 
          legend.position.inside = c(0.15, 0.85), 
          # legend.box = element_blank(),
          legend.background = element_rect(fill = NA, colour = NA, inherit.blank = FALSE),
          legend.key = element_rect(fill = NULL, colour = NA)) + 
    xlab('Longitude') + ylab('Latitude')
  
  ggsave2(Figure4, file = paste0(reddir, './figs/RedRiver_Figure4.pdf'), width = 7.5, height = 6.5)
  
}


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
  
  ggsave(Fig5, filename = paste0(reddir,'figs/RedRiver_Figure5.pdf'),  width = 6, height = 6)
  
  
#### FIG 6 fk  ####
  
  iml.dat <- filter(iml.importance, feat_id == 'fk') %>% mutate(red_id = factor(red_id, levels = c('Bat','Red'), labels = c('Batrachospermum', 'All red algae')), feature = sub('fk_','',feature) %>% sub('_',' ',.) )
  
  Fig6 <- ggplot(iml.dat) + geom_boxplot(aes(y = imprange, x = feature), outliers = FALSE) + 
    facet_grid(cols = vars(red_id), scales = 'free_y')  +  
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none") + 
    labs(x = 'Feature', y = 'Scaled importance')
  
  ggsave(Fig6, filename = paste0(reddir,'figs/RedRiver_Figure6.pdf'),  width = 6, height = 6)
  
 
} # provides figs/Fig2.pdf (cmb imp), figs/Fig3.pdf (fk imp)


# cmb VERDICT: use BAT - fk, pk, st; RED - fk, st, pk; underdogs mk, ap
# fk : BAT - BOD, Temp, depth, TN; RED - BOD, TN, Depth; underdog TP
# pk: BAT - 309, 302, 306; RED -  306, 309, 307 underdogs 304, 3011, 3010
# st: BAT - limestone, gravel; RED - mud, gravel; underdog; clay


# feature impact tables ####

if(T){
  # starting point all_imp_mcl
  # step 1 iml.imp <- lapply(all_imp_mcl, '[[', 2) # legnth 3652; 12 Mb
  # step 2 iml.importance <- lapply(seq_along(iml.imp), function(i){data.table(iml.imp[[i]])[, mean(importance), by = feature ] %>% transmute(feature = feature, importance = V1, impmax = as.numeric(decostand(V1, method = 'max')), imprange = as.numeric(decostand(V1, method = 'range'))) %>% bind_cols(., bmr_a[i, red_id:batch_id]) }) %>% bind_rows() # 1.7 Mb object; 24764 x 9
  
  tmp <- iml.importance %>% filter(hyp_id != '0') %>% # filter out untuned models
    summarise(importance = mean(importance), impmax = mean(impmax), imprange = signif(mean(imprange), 2), .by = c(red_id, feat_id, feature,  mod_id)) # lumps mod_id
  tmp <- mutate(tmp, group =ifelse(red_id=='Bat', 'Batrachospermum','All red algae'), model = ifelse(mod_id == 'rg','rf','xgb'), feature = sub('^.*_','', feature))
  
  # table feature importance per feature group, subtables by mod_id
  feature_groups <- unique(tmp$feat_id)
  supl_tbl <- lapply(feature_groups, function(x){mx <- filter(tmp, feat_id == x);
  pivot_wider(dplyr::select(mx, group, feature, model, imprange), names_from = c(feature), values_from = imprange)})
  names(supl_tbl) <- feature_groups
  
  # save raw supplementary tables 
  lapply(seq_along(supl_tbl), function(i){gt(supl_tbl[[i]]) %>% gtsave(filename = paste0('./tables/',names(supl_tbl)[i],'.docx'))})
} # provides 6 supplementary tables

# best tables
# https://www.reddit.com/r/rstats/comments/auyiit/any_advice_on_packages_for_neat_easy_tables/?rdt=40078
#  "xtable" and "stargazer".gt, knitr kableExtra
# https://www.reddit.com/r/Rlanguage/comments/am5549/library_for_creating_stylish_tables/


# ale and pdp plots ####

# cherry-pick 5 ale figs:
# Depth_rg_Red;
# BOD_rg_Red(+Bat) (free_y)
# Temp_rg_Red+Bat (free_y)
# TN_rg_Red
# O2p_rg_Red

bmr_a

# select
idx <- which(bmr_a$feat_id == 'fk'& bmr_a$red_id == 'Red' & bmr_a$mod_id == 'rg' & bmr_a$batch_id != '0') # length 150 = 3 mod_id x 50 hyp-headers

# average
ale_ag_fig_base <- lapply(idx, function(i){ summarise(ale.imp[[i]], value = mean(value, na.rm = T), .by = c(feature, X)) %>% bind_cols(., bmr_a[i, red_id:batch_id]) }) %>% bind_rows() %>% mutate(hyp_id = factor(hyp_id, levels = c('D','1','2'), labels = c('Default','rbv1','rbv2') )) # 105450; 6 Mb

# Figure 7 ALE plot ####

# pick a set of example hydrological - hydrochemical features
# ale_ag_fig_base$feature %>% unique() # to see the features
feats <- c('fk_BOD', 'fk_Depth','fk_TN','fk_O2p')
featnames <- c('Biological oxygen demand', 'Depth','Total nitrogen','Oxygen saturation')

ale_fig_lst <- X <-  list()
for(i in 1:length(feats)){
  X[[i]] <- filter(ale_ag_fig_base, feature == feats[i])$X %>% unique()
  rug_base <- data.table(X = unlist(dplyr::select(super_task, feats[i])), y = 0)
  
  ale_fig_lst[[i]] <- ggplot(filter(ale_ag_fig_base, feature == feats[i]), aes(x = X, y = value, col = hyp_id )) + 
    geom_line(aes(group = paste0(batch_id, hyp_id)), alpha = .05) +
    theme(legend.position="none") +
    geom_rug(mapping = aes(x = X, y = y), data = rug_base, sides = 'b', inherit.aes = F, col = rgb(.5,0,0, alpha = .1), length = unit(0.04, "npc")) +
    geom_xsidedensity(data = rug_base, mapping = aes(y = after_stat(density)), position = "stack", outline.type = 'upper',  col = 'black') + 
    ggside(x.pos = "bottom") +
    #theme(axis.text.x = element_text(angle = 90, vjust = .5))
    theme(ggside.axis.text.y = element_blank()) +
    labs(title = featnames[i], x = 'Feature', y = 'Relative importance')
}

# add color legend to last panel
ale_fig_lst[[i]] <- ale_fig_lst[[i]] + 
  theme(legend.position = c(0.9, 0.4), legend.background = element_rect(fill = NA), legend.key = element_rect(fill = NA, color = NA)) + 
  guides(colour = guide_legend(override.aes = list(linewidth = 1, alpha = 1))) +
  labs(color = 'Tuning spaces')


plot_grid(plotlist = lapply(seq_along(ale_fig_lst), function(i){ale_fig_lst[[i]] + xlim(quantile(X[[i]],  probs=c(0.05, .95)))}), ncol = 1) %>% 
  ggsave(filename = paste0(reddir,'figs/RedRiver_xgb_red_Figure7.pdf'), width = 6, height = 14)

plot_grid(plotlist = lapply(seq_along(ale_fig_lst), function(i){ale_fig_lst[[i]] + xlim(quantile(X[[i]], 
probs=c(0.05, .95)))}), ncol = 2) %>% 
  ggsave(filename = paste0(reddir,'figs/RedRiver_pres_Figure7.pdf'))


# 



feats <- c('fk_TN', 'fk_TP','fk_NO3','fk_NH4') # S1
featnames <- c('Total nitrogen', 'Total Phosphorus','Nitrate','Ammonium') # S1

feats <- c('fk_EC', 'fk_pH','fk_Flow_rate','fk_Flow_velocity') # S1
featnames <- c('Conductivity', 'pH','Flow Rate','Flow Velocity') # S1


ale_fig_lst <- X <-  list()
for(i in 1:length(feats)){
  X[[i]] <- filter(ale_ag_fig_base, feature == feats[i])$X %>% unique()
  rug_base <- data.table(X = unlist(dplyr::select(super_task, feats[i])), y = 0)
  
  ale_fig_lst[[i]] <- ggplot(filter(ale_ag_fig_base, feature == feats[i]), aes(x = X, y = value, col = hyp_id )) + 
    geom_line(aes(group = paste0(batch_id, hyp_id)), alpha = .05) +
    theme(legend.position="none") +
    geom_rug(mapping = aes(x = X, y = y), data = rug_base, sides = 'b', inherit.aes = F, col = rgb(.5,0,0, alpha = .1), length = unit(0.04, "npc")) +
    geom_xsidedensity(data = rug_base, mapping = aes(y = after_stat(density)), position = "stack", outline.type = 'upper',  col = 'black') + 
    ggside(x.pos = "bottom") +
    #theme(axis.text.x = element_text(angle = 90, vjust = .5))
    theme(ggside.axis.text.y = element_blank()) +
    labs(title = featnames[i], x = 'Feature', y = 'Relative importance')
}

# add color legend to last panel
ale_fig_lst[[i]] <- ale_fig_lst[[i]] + 
  theme(legend.position = c(0.9, 0.4), legend.background = element_rect(fill = NA), legend.key = element_rect(fill = NA, color = NA)) + 
  guides(colour = guide_legend(override.aes = list(linewidth = 1, alpha = 1))) +
  labs(color = 'Tuning spaces')


plot_grid(plotlist = lapply(seq_along(ale_fig_lst), function(i){ale_fig_lst[[i]] + xlim(quantile(X[[i]],  probs=c(0.05, .95)))}), ncol = 1) %>% 
  ggsave(filename = paste0(reddir,'figs/RedRiver_xgb_red_Figure7.pdf'), width = 6, height = 14)

plot_grid(plotlist = lapply(seq_along(ale_fig_lst), function(i){ale_fig_lst[[i]] + xlim(quantile(X[[i]], probs=c(0.05, .95)))}), ncol = 2) %>% 
  ggsave(filename = paste0(reddir,'figs/RedRiver_pres_FigureS2.pdf'))



# VOL 2 ####

# load 2nd all_imp_mcl - all_imp_mcl
load(file = paste0(reddir, 'dat/all_imp_mcl_v2.rda')) # length 3624

load(file = paste0(reddir, 'dat/all_imp_mcl_v3.rda'))

iml.imp <- lapply(all_imp_mcl, '[[', 1) # 5.6 Mb
ale.imp <- lapply(all_imp_mcl, '[[', 2) # 58 Mb
rm(all_imp_mcl)


# VOL 2 feature impact boxplots ####

if(T){
  
  # scale iml importance estimates with decostand, max and total
  
  iml.importance <- lapply(seq_along(iml.imp), function(i){data.table(iml.imp[[i]]) %>% 
      transmute(feature = feature, importance = importance, impmax = as.numeric(decostand(importance, method = 'max')), imprange = as.numeric(decostand(importance, method = 'range'))) %>%  bind_cols(., bmr_a[i, red_id:batch_id]) }) %>% bind_rows() # 1.7 Mb object; 24764 x 9
  
  
  #### FIG 5 cmb  ####
  
  iml.dat <- filter(iml.importance, feat_id == 'cmb', hyp_id != '0') %>% mutate(red_id = factor(red_id, levels = c('Bat','Red'), labels = c('Batrachospermum', 'All red algae')))
  
  Fig5 <- ggplot(iml.dat) + geom_boxplot(aes(y = imprange, x = feature), outliers = FALSE) + 
    facet_grid(cols = vars(red_id), scales = 'free_y')  +  
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none") + 
    labs(x = 'Feature group', y = 'Scaled importance') +
    scale_x_discrete(labels=c("ap" = "Bedrock", "fk" = "Hydro Chem", "mk" = "Land use", 'pk' = 'Land cover','st' = 'Substrate'))
  
  ggsave(Fig5, filename = paste0(reddir,'figs/RedRiver_Figure5_v3.pdf'),  width = 6, height = 6)
  
  
  #### FIG 6 fk  ####
  
  iml.dat <- filter(iml.importance, feat_id == 'fk') %>% mutate(red_id = factor(red_id, levels = c('Bat','Red'), labels = c('Batrachospermum', 'All red algae')), feature = sub('fk_','',feature) %>% sub('_',' ',.) )
  
  Fig6 <- ggplot(iml.dat) + geom_boxplot(aes(y = imprange, x = feature), outliers = FALSE) + 
    facet_grid(cols = vars(red_id), scales = 'free_y')  +  
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + theme(legend.position="none") + 
    labs(x = 'Feature', y = 'Scaled importance')
  
  ggsave(Fig6, filename = paste0(reddir,'figs/RedRiver_Figure6_vol3.pdf'),  width = 6, height = 6)
  
  
} # provides figs/RedRiver_Figure5_v2.pdf (cmb imp), figs/RedRiver_Figure6_v2.pdf (fk imp)

# cmb VERDICT: use BAT - fk, pk, st; RED - fk, st, pk; underdogs mk, ap
# fk : BAT - BOD, Temp, depth, TN; RED - BOD, TN, Depth; underdog TP
# pk: BAT - 309, 302, 306; RED -  306, 309, 307 underdogs 304, 3011, 3010
# st: BAT - limestone, gravel; RED - mud, gravel; underdog; clay


# ALE plots ####

super_task %>% dplyr::select(starts_with('fk_')) %>%  cor(use = 'pairwise.complete.obs')

ale.df <- lapply(seq_along(ale.imp), function(i){bind_cols(ale.imp[[i]], bmr_a[i,red_id:batch_id])} ) %>% bind_rows() %>% filter(batch_id != '0') %>% mutate(hyp_id = factor(hyp_id, levels = c('D','1','2'), labels = c('Default','rbv1','rbv2') )) 

# pick a set of features
# ale.df$feature %>% unique() # to see the features

## Fig 7 ####
feats <- c('fk_BOD', 'fk_Depth','fk_TN','fk_TP') # best features according to permutational feature importance
featnames <- c('Biological oxygen demand', 'Depth','Total nitrogen','Total phosphorus')
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

# add color legend to last panel
ale_fig_lst[[i]] <- ale_fig_lst[[i]] + 
  theme(legend.position = c(0.8, 0.8), legend.background = element_rect(fill = NA), legend.key = element_rect(fill = NA, color = NA)) + 
  guides(colour = guide_legend(override.aes = list(linewidth = 1, alpha = 1))) +
  labs(color = 'Tuning spaces')

# ale_fig_lst[[4]]

plot_grid(plotlist = lapply(seq_along(ale_fig_lst), function(i){ale_fig_lst[[i]] + xlim(quantile(X[[i]],  probs=c(0.05, .95)))}), ncol = 2) %>% 
  ggsave(filename = paste0(reddir,'figs/RedRiver_Figure7_vol4.pdf'), width = 8, height = 8)


  ## Supp Fig 1 ####

feats <- c('fk_NO3', 'fk_EC','fk_pH','fk_Flow_velocity')
featnames <- c('NO3', 'Conductivity','pH','Flow velocity')

ale_fig_lst <- X <-  list()
for(i in 1:length(feats)){
  X[[i]] <- filter(ale_ag_fig_base, feature == feats[i])$X %>% unique()
  rug_base <- data.table(X = unlist(dplyr::select(super_task, feats[i])), y = 0)
  
  ale_fig_lst[[i]] <- ggplot(filter(ale_ag_fig_base, feature == feats[i]), aes(x = X, y = value, col = hyp_id )) + 
    geom_line(aes(group = paste0(batch_id, hyp_id)), alpha = .05) +
    theme(legend.position="none") +
    geom_rug(mapping = aes(x = X, y = y), data = rug_base, sides = 'b', inherit.aes = F, col = rgb(.5,0,0, alpha = .1), length = unit(0.04, "npc")) +
    geom_xsidedensity(data = rug_base, mapping = aes(y = after_stat(density)), position = "stack", outline.type = 'upper',  col = 'black') + 
    ggside(x.pos = "bottom") +
    #theme(axis.text.x = element_text(angle = 90, vjust = .5))
    theme(ggside.axis.text.y = element_blank()) +
    labs(title = featnames[i], x = 'Feature', y = 'Relative importance')
}

# add color legend to last panel
ale_fig_lst[[i]] <- ale_fig_lst[[i]] + 
  theme(legend.position = c(0.8, 0.8), legend.background = element_rect(fill = NA), legend.key = element_rect(fill = NA, color = NA)) + 
  guides(colour = guide_legend(override.aes = list(linewidth = 1, alpha = 1))) +
  labs(color = 'Tuning spaces')

ale_fig_lst[[4]]

plot_grid(plotlist = lapply(seq_along(ale_fig_lst), function(i){ale_fig_lst[[i]] + xlim(quantile(X[[i]],  probs=c(0.05, .95)))}), ncol = 2) %>% 
  ggsave(filename = paste0(reddir,'figs/RedRiver_rg_red_Figure7_vol3_sup1.pdf'), width = 8, height = 8)



