library(gridExtra)
library(grid)

#VISUALIZE TABLE OF RESULTS#########################################################################################################################

val_df <- setNames(data.frame(matrix(ncol = 13, nrow = 9)), 
                   c('Inertia','Silhouette','Calinski-Harabaz','Accuracy','Precision','Recall','F1-Score','AMI','Adj_Rand','NMI','Homogeneity','Completeness','V_Measure'))

rownames(val_df) <- c('KNN','TS_K-Neighbors','Shapelet Classifier','K-Means++','TS_K-Means','K-Shape','LSTM_AE_K-Means++','LSTM_AE_TS_K-Means','LSTM_AE_K-Shape')

val_df[,1] <- c('--','--','--',1764.23,1.073,0.029,1898.12,2.96,0.0076)

val_df[,2] <- c('--','--','--',0.22,0.19,0.26,0.59,0.47,0.48)

val_df[,3] <- c('--','--','--',42.56,41.03,26.52,116.24,105.50,74.57)

val_df[,4] <- c(0.73,0.70,0.79,'--','--','--','--','--','--')

val_df[,5] <- c(0.76,0.78,0.86,'--','--','--','--','--','--')

val_df[,6] <- c(0.73,0.70,0.79,'--','--','--','--','--','--')

val_df[,7] <- c(0.72,0.65,0.80,'--','--','--','--','--','--')

val_df[,8] <- c(0.50,0.48,0.65,'--','--','--','--','--','--')

val_df[,9] <- c(0.49,0.45,0.61,'--','--','--','--','--','--')

val_df[,10] <- c(0.63,0.64,0.74,'--','--','--','--','--','--')

val_df[,11] <- c(0.60,0.58,0.75,'--','--','--','--','--','--')

val_df[,12] <- c(0.66,0.70,0.73,'--','--','--','--','--','--')

val_df[,13] <- c(0.63,0.63,0.74,'--','--','--','--','--','--')

#Create grid of results values
val_tbl <- tableGrob(val_df)

#Helper function to find cells in grid
find_cell <- function(table, row, col, name="core-fg"){
  l <- table$layout
  which(l$t==row & l$l==col & l$name==name)
}


#Coloring cells of best scores
ind <- find_cell(val_tbl, 4, 5, "core-bg")
val_tbl$grobs[ind][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)


ind2 <- find_cell(val_tbl, 4, 6, "core-bg")
val_tbl$grobs[ind2][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)

ind3 <- find_cell(val_tbl, 4, 7, "core-bg")
val_tbl$grobs[ind3][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)

ind4 <- find_cell(val_tbl, 4, 8, "core-bg")
val_tbl$grobs[ind4][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)

ind5 <- find_cell(val_tbl, 4, 9, "core-bg")
val_tbl$grobs[ind5][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)


ind6 <- find_cell(val_tbl, 4, 10, "core-bg")
val_tbl$grobs[ind6][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)

ind7 <- find_cell(val_tbl, 4, 11, "core-bg")
val_tbl$grobs[ind7][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)

ind8 <- find_cell(val_tbl, 4, 12, "core-bg")
val_tbl$grobs[ind8][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)

ind9 <- find_cell(val_tbl, 4, 13, "core-bg")
val_tbl$grobs[ind9][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)

ind10 <- find_cell(val_tbl, 4, 14, "core-bg")
val_tbl$grobs[ind10][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)

ind11 <- find_cell(val_tbl, 10, 2, "core-bg")
val_tbl$grobs[ind11][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)

ind12 <- find_cell(val_tbl, 8, 3, "core-bg")
val_tbl$grobs[ind12][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)

ind13 <- find_cell(val_tbl, 8, 4, "core-bg")
val_tbl$grobs[ind13][[1]][["gp"]] <- gpar(fill="darkolivegreen1", col = "darkolivegreen4", lwd=5)


grid.draw(val_tbl)