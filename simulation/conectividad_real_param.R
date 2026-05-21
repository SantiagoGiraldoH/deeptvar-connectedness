###### GRÁFICOS DE CONECTIVIDAD CON PARÁMETROS REALES TV-VAR(2)

# Instalar y cargar paquetes necesarios
# install.packages(c("ggplot2", "tidyr", "dplyr", "igraph"))
library(ggplot2)
library(tidyr)
library(dplyr)
library(igraph)


# LECTURA DE ARCHIVOS Y TRANSFORMACIÓN A ARRAYS 3D

df_A1 <- read.csv("A1_true.csv")
df_A2 <- read.csv("A2_true.csv")
df_Sigma <- read.csv("Sigma_true.csv")
Y <- read.csv("serie_simulada_con_indice.csv")
# Parámetros iniciales
T <- nrow(df_A1)   # Número de observaciones (ej. 365)
k <- 4             # Número de variables
p <- 2             # Lags
H <- 12            # Horizonte de predicción
var_names <- paste0("V", 1:k) # Nombres genéricos V1, V2, V3, V4

# Inicializar los arrays 3D vacíos
A1_estimates <- array(0, dim = c(T, k, k))
A2_estimates <- array(0, dim = c(T, k, k))
Sigma_estimates <- array(0, dim = c(T, k, k))

# Matriz Sigma constante basada en la exportación previa (4x4)
Sigma_static <- as.matrix(df_Sigma)

# Llenar los arrays
for(t in 1:T) {

  A1_estimates[t, , ] <- matrix(as.numeric(unlist(df_A1[t, ])), nrow = k, ncol = k, byrow = TRUE)
  A2_estimates[t, , ] <- matrix(as.numeric(unlist(df_A2[t, ])), nrow = k, ncol = k, byrow = TRUE)
  
  # Reconstruir Sigma_t a partir de los 10 elementos de la triangular inferior
  sigma_row <- as.numeric(unlist(df_Sigma[t, ]))
  mat_sigma <- matrix(0, nrow = k, ncol = k)
  
  idx <- 1
  for (i in 1:k) {
    for (j in 1:i) {
      mat_sigma[i, j] <- sigma_row[idx]
      mat_sigma[j, i] <- sigma_row[idx] 
      idx <- idx + 1
    }
  }
  
  Sigma_estimates[t, , ] <- mat_sigma
}

cat("Dimensiones de A1:", paste(dim(A1_estimates), collapse=" x "), "\n")
cat("Dimensiones de Sigma:", paste(dim(Sigma_estimates), collapse=" x "), "\n")


# CÁLCULO DE LA DESCOMPOSICIÓN DE VARIANZA (GFEVD)

gfevd_array <- array(0, dim = c(T, k, k))

for (t in 1:T) {
  A1 <- A1_estimates[t, , ]
  A2 <- A2_estimates[t, , ]
  Sigma <- Sigma_estimates[t, , ]
  
  # Construir matriz compañera M_t para un VAR(2)
  M_t <- rbind(
    cbind(A1, A2),
    cbind(diag(k), matrix(0, k, k))
  )
  
  # Respuestas a Impulsos Ortogonalizados (VMA representation)
  B <- array(0, dim = c(H, k, k))
  B[1, , ] <- diag(k)
  
  M_temp <- M_t
  for (h in 2:H) {
    B[h, , ] <- M_temp[1:k, 1:k]
    M_temp <- M_temp %*% M_t 
  }
  
  # (GIRF)
  num <- matrix(0, k, k)
  
  for (h in 1:H) {
    girf_base <- B[h, , ] %*% Sigma
    for (j in 1:k) {
      psi_j <- girf_base[, j] / sqrt(Sigma[j, j])
      num[, j] <- num[, j] + (psi_j^2)
    }
  }
  
  # Normalizar la matriz de varianza (las filas suman 1)
  den <- rowSums(num)
  gfevd_t <- sweep(num, 1, den, FUN = "/")
  
  gfevd_array[t, , ] <- gfevd_t
}


# CÁLCULO DE MEDIDAS DE CONECTIVIDAD DINÁMICA

TCI  <- numeric(T)
TO   <- matrix(0, nrow = T, ncol = k, dimnames = list(NULL, var_names))
FROM <- matrix(0, nrow = T, ncol = k, dimnames = list(NULL, var_names))
NET  <- matrix(0, nrow = T, ncol = k, dimnames = list(NULL, var_names))
NPDC <- array(0, dim = c(T, k, k))

for (t in 1:T) {
  gfevd_t <- gfevd_array[t, , ]
  
  # Índice de Conectividad Total (TCI)
  TCI[t] <- (sum(gfevd_t) - sum(diag(gfevd_t))) / k * 100
  
  for (i in 1:k) {
    FROM[t, i] <- (sum(gfevd_t[i, ]) - gfevd_t[i, i]) * 100
    TO[t, i]   <- (sum(gfevd_t[, i]) - gfevd_t[i, i]) * 100
    NET[t, i]  <- TO[t, i] - FROM[t, i]
    
    for (j in 1:k) {
      NPDC[t, i, j] <- (gfevd_t[j, i] - gfevd_t[i, j]) * 100
    }
  }
}


# GRÁFICOS Y VISUALIZACIONES

dates <- seq(as.Date("2020-01-01"), by = "days", length.out = T)


# Gráfico 1: Índice de Conectividad Total (TCI)

df_tci <- data.frame(Date = dates, TCI = TCI)

p1 <- ggplot(df_tci, aes(x = Date, y = TCI)) +
  geom_area(fill = "#4A90E2", alpha = 0.3) +
  geom_line(color = "#003366", linewidth = 1) +
  theme_minimal(base_size = 14) +
  labs(title = "Índice de Conectividad Total (TCI)",
       subtitle = "Evolución dinámica a lo largo del tiempo",
       x = "Fecha",
       y = "TCI (%)") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
        panel.grid.minor = element_blank())
print(p1)


# Gráfico 2: Conectividad Direccional Neta por Variable

df_net <- as.data.frame(NET)
df_net$Date <- dates

df_net_long <- df_net %>%
  pivot_longer(cols = all_of(var_names), names_to = "Variable", values_to = "NetValue") %>%
  mutate(Role = ifelse(NetValue > 0, "Transmisor Neto", "Receptor Neto"))

p2 <- ggplot(df_net_long, aes(x = Date, y = NetValue, fill = Role)) +
  geom_col(width = 1.2, alpha = 0.85, position = "identity") +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  facet_wrap(~ Variable, scales = "fixed", ncol = 2) +
  scale_fill_manual(values = c("Transmisor Neto" = "#2ca02c", "Receptor Neto" = "#d62728")) +
  theme_minimal(base_size = 14) +
  labs(title = "Conectividad Direccional Neta por Variable",
       x = "Fecha",
       y = "Conectividad Neta (%)",
       fill = "Rol:") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid.minor = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face = "bold", size = 12),
        strip.background = element_rect(fill = "gray90", color = NA))
print(p2)


# Gráfico 3: Red de Conectividad Neta por Pares (NPDC Promedio)

avg_NPDC <- apply(NPDC, c(2, 3), mean)
rownames(avg_NPDC) <- var_names
colnames(avg_NPDC) <- var_names
adj_matrix <- pmax(avg_NPDC, 0)

# Crear grafo
net_graph <- graph_from_adjacency_matrix(adj_matrix, mode = "directed", weighted = TRUE)

E(net_graph)$width <- E(net_graph)$weight * 1.5 

avg_NET_abs <- colMeans(abs(NET))
V(net_graph)$size <- (avg_NET_abs * 2) + 20 

avg_NET_raw <- colMeans(NET)
V(net_graph)$color <- ifelse(avg_NET_raw > 0, "#D35400", "#7F8C8D")
V(net_graph)$frame.color <- ifelse(avg_NET_raw > 0, "#D35400", "#7F8C8D")

# Configurar visualización de la red
par(mar = c(2, 2, 3, 2)) # Ajustar márgenes

plot(net_graph, 
     layout = layout_in_circle(net_graph),
     edge.arrow.size = 0.6,
     edge.color = "gray30",
     edge.curved = 0.15,
     vertex.label.color = "black",
     vertex.label.font = 2,
     vertex.label.cex = 1.2,
     main = "Red de Conectividad Neta Promedio (NPDC)")
