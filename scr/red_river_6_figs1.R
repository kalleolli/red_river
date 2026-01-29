# providers map figures 1-4
# no prior calculations, takes data from dat0 folder

# discover working directory
# assumes red_river.Rproj  in the  ~/Documents hierarchy

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


reddir <- list.files(path = '~/Documents', full.names = TRUE, recursive = TRUE, pattern = 'red_river.Rproj') %>% dirname()

load(paste0(reddir, '/dat0/est.rda')) # est 40 kB
load(paste0(reddir, '/dat0/red_rivers.rda')) # sampled rivers sf for figures
load(paste0(reddir, '/dat0/super_task.rda')) # mlr3 tasks prototype

if(T){
  
  basemap <- ggplot(data = est) +
    geom_sf(fill= "whitesmoke") + 
    geom_sf(data = red_rivers, aes(geometry = geometry), colour = 'gray', lwd = 0.1) + 
    geom_point(data = unique(dplyr::select(super_task, X, Y)), aes(x=X, y=Y), color = 'darkgray', size = 1 ) + 
    annotation_scale(location = "bl", width_hint = 0.2) + xlab('Longitude') + ylab('Latitude') + 
    annotation_north_arrow(location = "bl", which_north = "true", pad_x = unit(0.75, "in"), pad_y = unit(0.5, "in"), style = north_arrow_fancy_orienteering) + theme(panel.background = element_rect(fill = "aliceblue"))
  
  
  # insert map for northern Europe
  neurope <- ne_countries( scale = "medium", returnclass = "sf") %>% st_crop(., xmin = 2, xmax = 30, ymin = 50, ymax = 70)
  
  ggm1 <- ggplot(data = neurope) + 
    geom_sf(fill = 'whitesmoke') + 
    geom_sf(data = st_as_sfc(st_bbox(st_transform(est, 4326))), fill = NA, color = 'red', linewidth = .3 ) +
    ggspatial::coord_sf(xlim=c(2, 30), ylim = c(55, 70), expand = T) + theme_void()
  
  
  
  ## Figure 1 ####
  
  Figure1 <- cowplot::ggdraw() +
    draw_plot(basemap) +
    draw_plot(ggm1, scale = 0.2, halign = 0.08, valign = 0.82)
  
  ggsave2(Figure1, file = paste0(reddir, '/figs/red_river_fig1.pdf'), width = 7.5, height = 6.5)
  
  
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
  
  ggsave2(Figure2, file = paste0(reddir, '/figs/red_river_fig2.pdf'), width = 7.5, height = 6.5)
  
  
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
  
  ggsave2(Figure3, file = paste0(reddir, '/figs/red_river_fig3.pdf'), width = 7.5, height = 6.5)
  
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
  
  ggsave2(Figure4, file = paste0(reddir, '/figs/red_river_fig4.pdf'), width = 7.5, height = 6.5)
  
}

