setwd("C:\\Users\\Daniel Atzberger\\Documents\\IEEE_Vis24\\Git Repos\\Vis24_DR_Stability\\Experiments\\data_final")

library(ggplot2)
df <- read.csv('layout12.csv')

my_colors <- c("#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f")
# my_colors <- c("#547980", "#9DE0AD", "#A7226E", "#EC2049", "#F26B38", "#F7DB4F", "#2F9599")


ggplot(df, aes(x = x, y = y, color = factor(classes))) +
  geom_point(size = 2.0) +
  scale_color_manual(values = my_colors) +  # Set custom color scheme
  theme(panel.grid.major = element_blank(),
        panel.background = element_blank(),
        legend.position = "none",
        axis.ticks.length = unit(0, "pt"),
        axis.text.x = element_text(size = 0),
        axis.text.y = element_text(size = 0)) + 
  labs(x="", y="")


ggplot(df, aes(x = x_new, y = y_new, color = factor(classes))) +
  geom_point(size = 2.0) +
  scale_color_manual(values = my_colors) +  # Set custom color scheme
  theme(panel.grid.major = element_blank(),
        panel.background = element_blank(),
        legend.position = "none",
        axis.ticks.length = unit(0, "pt"),
        axis.text.x = element_text(size = 0),
        axis.text.y = element_text(size = 0)) + 
  labs(x="", y="")
