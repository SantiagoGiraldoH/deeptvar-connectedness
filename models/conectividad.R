# MEDIDAS DE CONECTIVIDAD

library(zoo)
library(devtools)
library(ConnectednessApproach)


# Obtener las medidas de conectividad desde el DeepTVAR 

#install.packages("reticulate")
library(reticulate)

np <- import("numpy")

A1_estimates <- np$load("A1_empirical (1).npy")
A2_estimates <- np$load("A2_empirical (1).npy")
Sigma_estimates <- np$load("Sigma_empirical (1).npy")

#Verificar las dimensiones (T, k, k)
print(dim(A1_estimates))
print(dim(Sigma_estimates))

T <- 1053
k <- 4
p <- 2
H <- 12 # Horizonte de predicción

# Iniciar un array vacío para guardar la GFEVD para cada timepo t

gfevd_array <- array(0, dim = c(T, k, k))

for (t in 1:T) {
  # Extraer matrices
  
  A1 <- A1_estimates[t, , ]
  A2 <- A2_estimates[t, , ]
  Sigma <- Sigma_estimates[t, , ]
  
  # Construir matriz M_t
  M_t <- rbind(
    cbind(A1, A2),
    cbind(diag(k), matrix(0, k, k))
  )
  
  # Iniciar coeficientes B_h
  B <- array(0, dim = c(H, k, k))
  B[1, , ] <- diag(k) # B_0 es la identidad
  
  M_temp <- M_t
  for (h in 2:H) {
    
    B[h, , ] <- M_temp[1:k, 1:k]
    M_temp <- M_temp %*% M_t 
  }
  
  
  num <- matrix(0, k, k)
  
  for (h in 1:H) {
    # GIRF matriz: B_h * Sigma
    
    girf_base <- B[h, , ] %*% Sigma
    
    for (j in 1:k) {
      
      # Dividir por la desviación estandar de la variable 'j'
      psi_j <- girf_base[, j] / sqrt(Sigma[j, j])
      
      # Eleva al cuadrado y se agrega a la suma acumulada
      num[, j] <- num[, j] + (psi_j^2)
    }
  }
  
  
  #suma por todas las variables trasmisoras j
  den <- rowSums(num)
  
  # Normalizar
  gfevd_t <- num / den
  
  #Matriz de conectividad
  gfevd_array[t, , ] <- gfevd_t
}


# Medidas de conectividad dinámica
TCI <- numeric(T)
TO <- matrix(0, nrow = T, ncol = k)
FROM <- matrix(0, nrow = T, ncol = k)
NET <- matrix(0, nrow = T, ncol = k)


NPDC <- array(0, dim = c(T, k, k))

for (t in 1:T) {
  
  gfevd_t <- gfevd_array[t, , ]
  
  
  # Agregar el índice total de la red
  
  off_diagonal_sum <- sum(gfevd_t) - sum(diag(gfevd_t))
  TCI[t] <- (off_diagonal_sum / k) * 100
  
  for (i in 1:k) {
    
    # onectividad Direccional Recibida
    
    FROM[t, i] <- (sum(gfevd_t[i, ]) - gfevd_t[i, i]) * 100
    
    # onectividad Direccional Transmitida
    
    TO[t, i] <- (sum(gfevd_t[, i]) - gfevd_t[i, i]) * 100
    
    # Conectividad Direccional Neta:
    NET[t, i] <- TO[t, i] - FROM[t, i]
    
    for (j in 1:k) {
      
      # Conectividad Neta por Pares
      
      NPDC[t, i, j] <- (gfevd_t[j, i] - gfevd_t[i, j]) * 100
    }
  }
}


var_names <- c("Oficina 1", "Oficina 2", "Oficina 3", "Oficina 4")
colnames(TO) <- var_names
colnames(FROM) <- var_names
colnames(NET) <- var_names


## Gráficas

#install.packages("igraph")

library(ggplot2)
library(tidyr)
library(dplyr)
library(igraph)

theme_paper <- theme_minimal(base_family = "serif") +
  theme(
    text              = element_text(family = "serif"),
    axis.line         = element_line(color = "black"),
    axis.ticks        = element_line(color = "black"),
    panel.grid.major  = element_line(color = "#EAEAF2"),
    panel.grid.minor  = element_blank(),
    panel.border      = element_rect(color = "black", fill = NA),
    plot.background   = element_rect(fill = "white", color = NA),
    panel.background  = element_rect(fill = "white", color = NA)
  )


dates <- seq(as.Date("2022-01-01"), by = "days", length.out = T)
var_names <- c("Oficina 1", "Oficina 2", "Oficina 3", "Oficina 4")

# Preparar data
df_tci <- data.frame(Date = dates, TCI = TCI)

# Plot
ggplot(df_tci, aes(x = Date, y = TCI)) +
  geom_area(fill = "blue", alpha = 0.6) +
  geom_line(color = "blue", linewidth = 0.5) +
  labs(
    x = "Tiempo",
    y = "TCI (%)"
  ) +
  theme_paper

df_net <- as.data.frame(NET)
colnames(df_net) <- var_names
df_net$Date <- dates

df_net_long <- df_net %>%
  pivot_longer(cols = all_of(var_names), names_to = "Variable", values_to = "NetValue") %>%
  mutate(Role = ifelse(NetValue > 0, "Transmisor", "Receptor"))


ggplot(df_net_long, aes(x = Date, y = NetValue, fill = Role)) +
  geom_col(
    width = 1,
    alpha = 0.85,
    position = "identity"   # ← clave: evita el apilado y colorea por signo
  ) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  facet_wrap(~ Variable, scales = "fixed") +
  scale_fill_manual(
    values = c(
      "Transmisor" = "#ff7f0e",
      "Receptor"  = "gray"
    )
  ) +
  theme_minimal(base_family = "serif") +
  labs(
    x     = "Tiempo",
    y     = "Conectividad Neta (%)",
    fill  = NULL             
  ) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),  
    legend.position  = "bottom",
    legend.key.size  = unit(0.5, "cm"),
    legend.text      = element_text(size = 11),
    strip.text       = element_text(size = 12, face = "bold")
  ) + theme_paper

avg_NPDC <- apply(NPDC, c(2, 3), mean)
rownames(avg_NPDC) <- var_names
colnames(avg_NPDC) <- var_names

adj_matrix <- pmax(avg_NPDC, 0)

net_graph <- graph_from_adjacency_matrix(adj_matrix, mode = "directed", weighted = TRUE)

E(net_graph)$width <- E(net_graph)$weight * 1.5 


avg_NET <- colMeans(abs(NET))
V(net_graph)$size <- avg_NET * 5 

avg_NET_raw <- colMeans(NET)
V(net_graph)$color <- ifelse(avg_NET_raw > 0, "tomato", "lightblue")


set.seed(42)
plot(net_graph, 
     layout = layout_in_circle(net_graph),
     edge.arrow.size = 0.8,
     edge.color = "gray40",
     edge.curved = 0.2, # Adds a nice curve to the arrows
     vertex.label.color = "black",
     vertex.label.family = "sans",
     vertex.label.cex = 1.2,
     vertex.frame.color = "gray20",
     main = "Average Net Pairwise Connectedness Network")


# Código por pares


library(igraph)

# Definir los índices de tiempo para cada período

idx_crisis <- which(dates >= "2022-01-01" & dates <= "2023-12-31")
idx_calma  <- which(dates >= "2024-01-01" & dates <= "2025-12-31")

# Crear una función para evitar duplicar código
generar_grafo_periodo <- function(NPDC_array, NET_matrix, indices, titulo) {
  
  # Filtrar los datos solo para el período seleccionado
  NPDC_filtrado <- NPDC_array[indices, , ]
  NET_filtrado <- NET_matrix[indices, ]
  
  # Promediar en la dimensión temporal (ahora restringida al período)
  avg_NPDC <- apply(NPDC_filtrado, c(2, 3), mean)
  rownames(avg_NPDC) <- var_names
  colnames(avg_NPDC) <- var_names
  
  # Matriz de adyacencia
  adj_matrix <- pmax(avg_NPDC, 0)
  
  # Construir el grafo
  net_graph <- graph_from_adjacency_matrix(adj_matrix, mode = "directed", weighted = TRUE)
  
  # Grosor de las aristas
  E(net_graph)$width <- E(net_graph)$weight * 1.5 
  
  avg_NET <- colMeans(abs(NET_filtrado))
  V(net_graph)$size <- avg_NET * 5 
  
  avg_NET_raw <- colMeans(NET_filtrado)
  V(net_graph)$color <- ifelse(avg_NET_raw > 0, "#D35400", "#7F8C8D")
  
  # Graficar
  set.seed(42) # Para que la posición de los nodos sea idéntica en ambos gráficos
  plot(net_graph, 
       layout = layout_in_circle(net_graph),
       edge.arrow.size = 0.8,
       edge.color = "gray40",
       edge.curved = 0.2, 
       vertex.label.color = "black",
       vertex.label.family = "sans",
       vertex.label.cex = 1.2,
       vertex.frame.color = "gray20")
}


par(mfrow = c(1, 2))

# Generar ambos grafos
generar_grafo_periodo(NPDC, NET, idx_crisis, "Red de Crisis (2022)")
generar_grafo_periodo(NPDC, NET, idx_calma, "Red de Estabilización (2023-2024)")

# Restaurar los parámetros gráficos a su valor original
par(mfrow = c(1, 1))
