
library(tvReg)
library(shrinkTVPVAR)
library(igraph)
library(ggplot2)
library(dplyr)
library(tidyr)

raw_df  <- read.csv("serie_simulada_con_indice.csv")


Y_mat <- as.matrix(raw_df[, grep("^V", names(raw_df))])

k       <- ncol(Y_mat)          # 4
p_lags  <- 2
H       <- 12                   # Horizonte de predicción para GFEVD
var_names <- colnames(Y_mat)    # "V1","V2","V3","V4"

cat(sprintf("Serie cargada: %d obs × %d variables\n", nrow(Y_mat), k))


# FUNCIÓN AUXILIAR: Cálculo de GFEVD dado A1, A2, Sigma

compute_gfevd <- function(A1, A2, Sigma, H, k) {
  # Construir la matriz compañera del VAR(2)
  M <- rbind(cbind(A1, A2),
             cbind(diag(k), matrix(0, k, k)))
  
  # Respuestas al impulso (representación VMA)
  B      <- array(0, dim = c(H, k, k))
  B[1,,] <- diag(k)
  M_pow  <- M
  for (h in 2:H) {
    B[h,,] <- M_pow[1:k, 1:k]
    M_pow  <- M_pow %*% M
  }
  
  # GIRF numerador
  num <- matrix(0, k, k)
  for (h in 1:H) {
    base <- B[h,,] %*% Sigma
    for (j in 1:k) {
      psi_j    <- base[, j] / sqrt(max(Sigma[j, j], 1e-12))
      num[, j] <- num[, j] + psi_j^2
    }
  }
  
  # Normalizar (filas suman 1)
  sweep(num, 1, rowSums(num), FUN = "/")
}

# FUNCIÓN AUXILIAR: Conectividad (NET, NPDC, TCI) desde array GFEVD

compute_connectedness <- function(gfevd_array, k, Te, var_names) {
  TCI  <- numeric(Te)
  TO   <- matrix(0, Te, k, dimnames = list(NULL, var_names))
  FROM <- matrix(0, Te, k, dimnames = list(NULL, var_names))
  NET  <- matrix(0, Te, k, dimnames = list(NULL, var_names))
  NPDC <- array(0, dim = c(Te, k, k))
  
  for (t in 1:Te) {
    g <- gfevd_array[t,,]
    TCI[t] <- (sum(g) - sum(diag(g))) / k * 100
    for (i in 1:k) {
      FROM[t, i] <- (sum(g[i,]) - g[i,i]) * 100
      TO[t, i]   <- (sum(g[,i]) - g[i,i]) * 100
      NET[t, i]  <- TO[t, i] - FROM[t, i]
      for (j in 1:k) NPDC[t, i, j] <- (g[j,i] - g[i,j]) * 100
    }
  }
  list(TCI = TCI, TO = TO, FROM = FROM, NET = NET, NPDC = NPDC)
}

# FUNCIÓN AUXILIAR: Gráfico de red NPDC

plot_net_graph <- function(NET, NPDC, var_names, title) {
  avg_NPDC <- apply(NPDC, c(2,3), mean)
  rownames(avg_NPDC) <- var_names
  colnames(avg_NPDC) <- var_names
  
  adj   <- pmax(avg_NPDC, 0)
  g_net <- graph_from_adjacency_matrix(adj, mode = "directed", weighted = TRUE)
  
  E(g_net)$width    <- E(g_net)$weight * 1.5
  avg_abs           <- colMeans(abs(NET))
  V(g_net)$size     <- (avg_abs * 2) + 20
  avg_raw           <- colMeans(NET)
  V(g_net)$color       <- ifelse(avg_raw > 0, "#D35400", "#7F8C8D")
  V(g_net)$frame.color <- ifelse(avg_raw > 0, "#D35400", "#7F8C8D")
  
  par(mar = c(2,2,3,2))
  set.seed(123)
  plot(g_net,
       layout            = layout_in_circle(g_net),
       edge.arrow.size   = 0.6,
       edge.color        = "gray30",
       edge.curved       = 0.15,
       vertex.label.color = "black",
       vertex.label.font = 2,
       vertex.label.cex  = 1.2,
       main              = title)
}

#  ESTIMACIÓN CON tvReg

set.seed(42)
cat("\n--- Estimando TV-VAR(2) con tvReg... ---\n")
tv_var_est <- tvVAR(Y_mat, p = p_lags, type = "const")


coef_list <- coef(tv_var_est)

# Verificar estructura
stopifnot(is.list(coef_list), length(coef_list) == k)
Te_tv  <- nrow(coef_list[[1]])
n_pred <- ncol(coef_list[[1]])  


# Inicializar arrays
A1_tv    <- array(0, dim = c(Te_tv, k, k))
A2_tv    <- array(0, dim = c(Te_tv, k, k))
Sigma_tv <- array(0, dim = c(Te_tv, k, k))

# Sigma empírica (constante) a partir de los residuos
Sigma_emp <- cov(residuals(tv_var_est))

# Extracción de A1 y A2
# Columnas: 1 = const, 2:(k+1) = lag 1, (k+2):(2k+1) = lag 2
for (t in 1:Te_tv) {
  for (i in 1:k) {
    A1_tv[t, i, ] <- coef_list[[i]][t, 2:(k + 1)]
    A2_tv[t, i, ] <- coef_list[[i]][t, (k + 2):(2*k + 1)]
  }
  Sigma_tv[t,,] <- Sigma_emp
}

cat("Extracción tvReg completada.\n")

# GFEVD
gfevd_tv <- array(0, dim = c(Te_tv, k, k))
for (t in 1:Te_tv)
  gfevd_tv[t,,] <- compute_gfevd(A1_tv[t,,], A2_tv[t,,], Sigma_tv[t,,], H, k)

# Medidas de conectividad
conn_tv <- compute_connectedness(gfevd_tv, k, Te_tv, var_names)

# Gráfico TCI
dates_tv <- seq(as.Date("2020-01-01"), by = "days", length.out = Te_tv)
df_tci_tv <- data.frame(Date = dates_tv, TCI = conn_tv$TCI)
p_tci <- ggplot(df_tci_tv, aes(x = Date, y = TCI)) +
  geom_area(fill = "#4A90E2", alpha = 0.3) +
  geom_line(color = "#003366", linewidth = 1) +
  theme_minimal(base_size = 14) +
  labs(title = "TCI – Estimación tvReg", x = "Fecha", y = "TCI (%)") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid.minor = element_blank())
print(p_tci)

# Gráfico 
df_net_tv <- as.data.frame(conn_tv$NET)
df_net_tv$Date <- dates_tv
df_net_long <- pivot_longer(df_net_tv, all_of(var_names),
                            names_to = "Variable", values_to = "NetValue") %>%
  mutate(Role = ifelse(NetValue > 0, "Transmisor Neto", "Receptor Neto"))

p_net <- ggplot(df_net_long, aes(x = Date, y = NetValue, fill = Role)) +
  geom_col(width = 1.2, alpha = 0.85, position = "identity") +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  facet_wrap(~ Variable, scales = "fixed", ncol = 2) +
  scale_fill_manual(values = c("Transmisor Neto" = "#D35400",
                               "Receptor Neto"   = "#7F8C8D")) +
  theme_minimal(base_size = 14) +
  labs(title = "Conectividad Neta por Variable – tvReg",
       x = "Fecha", y = "Conectividad Neta (%)", fill = "Rol:") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid.minor = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face = "bold", size = 12),
        strip.background = element_rect(fill = "gray90", color = NA))
print(p_net)

# Gráfico de red
plot_net_graph(conn_tv$NET, conn_tv$NPDC, var_names,
               "Red de Conectividad Neta (Estimada con tvReg)")



# ESTIMACIÓN CON shrinkTVPVAR

cat("\n--- Estimando TV-VAR(2) con shrinkTVPVAR... ---\n")
set.seed(42)
tvp_mod <- shrinkTVPVAR(
  Y_mat,
  p      = p_lags,
  const  = TRUE,
  niter  = 2000,
  nburn  = 1000
)

Te_sh   <- nrow(Y_mat) - p_lags   # observaciones efectivas (500-2=498)
n_draws <- tvp_mod$niter - tvp_mod$nburn


cat("\n--- Estructura de tvp_mod$beta ---\n")
cat("  clase   :", class(tvp_mod$beta), "\n")
cat("  longitud:", length(tvp_mod$beta), "\n")
if (is.list(tvp_mod$beta)) {
  cat("  clase beta[[1]]     :", class(tvp_mod$beta[[1]]), "\n")
  if (is.list(tvp_mod$beta[[1]])) {
    cat("  longitud beta[[1]]  :", length(tvp_mod$beta[[1]]), "\n")
    cat("  dim beta[[1]][[1]]  :", paste(dim(tvp_mod$beta[[1]][[1]]), collapse="×"), "\n")
  } else {
    cat("  dim beta[[1]]       :", paste(dim(tvp_mod$beta[[1]]), collapse="×"), "\n")
  }
} else {
  cat("  dim(beta)           :", paste(dim(tvp_mod$beta), collapse="×"), "\n")
}

cat("\n--- Estructura de tvp_mod$Sigma ---\n")
cat("  clase:", class(tvp_mod$Sigma), "\n")
cat("  dim  :", paste(dim(tvp_mod$Sigma), collapse="×"), "\n")

# Inicializar arrays de estimaciones puntuales
A1_sh    <- array(0, dim = c(Te_sh, k, k))
A2_sh    <- array(0, dim = c(Te_sh, k, k))
Sigma_sh <- array(0, dim = c(Te_sh, k, k))


# EXTRACCIÓN DE BETA

# Colapsar la dimensión MCMC (dim 5) promediando → [k, k, p, Te]
b_mean <- apply(tvp_mod$beta, 1:4, mean)  # [k_eq, k_var, p, T]

for (t in 1:Te_sh) {
  A1_sh[t, , ] <- b_mean[, , 1, t]   # lag 1: beta[i, j, lag=1, t]
  A2_sh[t, , ] <- b_mean[, , 2, t]   # lag 2: beta[i, j, lag=2, t]
}

cat("Beta extraído correctamente: array [k, k, p, T, n_draws] → medias posteriores\n")


# EXTRACCIÓN DE SIGMA 

sigma_ok <- FALSE
sigma_dims <- dim(tvp_mod$Sigma)

n_sigma_dims <- length(sigma_dims)

dim_matches <- function(dims, val) any(!is.na(dims) & dims == val)

if (n_sigma_dims == 4 &&
    dim_matches(sigma_dims, Te_sh) &&
    dim_matches(sigma_dims, n_draws)) {
  
  
  pos_T      <- which(sigma_dims == Te_sh)[1]
  pos_draws  <- which(sigma_dims == n_draws)[1]
  
  
  S_mean <- apply(tvp_mod$Sigma, (1:4)[-pos_draws], mean)
  
  s_dims_new <- sigma_dims[-pos_draws]
  pos_T_new  <- which(s_dims_new == Te_sh)[1]
  
  for (t in 1:Te_sh) {
    if (pos_T_new == 3) Sigma_sh[t,,] <- S_mean[,,t]
    if (pos_T_new == 2) Sigma_sh[t,,] <- S_mean[,t,]
    if (pos_T_new == 1) Sigma_sh[t,,] <- S_mean[t,,]
  }
  sigma_ok <- TRUE
  cat(sprintf("Sigma extraído: dim original [%s], promediado sobre dim %d\n",
              paste(sigma_dims, collapse="×"), pos_draws))
  
} else if (n_sigma_dims == 3 && dim_matches(sigma_dims, Te_sh)) {
  # Sin dimensión MCMC: ya son medias (p.ej. [k, k, T] o [T, k, k])
  pos_T <- which(sigma_dims == Te_sh)[1]
  for (t in 1:Te_sh) {
    if (pos_T == 3) Sigma_sh[t,,] <- tvp_mod$Sigma[,,t]
    if (pos_T == 1) Sigma_sh[t,,] <- tvp_mod$Sigma[t,,]
  }
  sigma_ok <- TRUE
  cat(sprintf("Sigma extraído: array 3D [%s] (sin dim MCMC)\n",
              paste(sigma_dims, collapse="×")))
  
} else if (is.list(tvp_mod$Sigma)) {
  
  if (length(tvp_mod$Sigma) == Te_sh && is.matrix(tvp_mod$Sigma[[1]])) {
    for (t in 1:Te_sh) Sigma_sh[t,,] <- tvp_mod$Sigma[[t]]
    sigma_ok <- TRUE
    cat("Sigma extraído: lista de", Te_sh, "matrices [k×k]\n")
  }
}

if (!sigma_ok) {
  Sigma_emp_sh <- cov(Y_mat[(p_lags + 1):nrow(Y_mat), ])
  for (t in 1:Te_sh) Sigma_sh[t,,] <- Sigma_emp_sh
}

# Asegurar simetría y positividad de cada Sigma_t
for (t in 1:Te_sh) {
  S <- Sigma_sh[t,,]
  S <- (S + t(S)) / 2
  eig <- eigen(S, symmetric = TRUE)
  if (any(eig$values < 1e-8)) {
    eig$values <- pmax(eig$values, 1e-8)
    S <- eig$vectors %*% diag(eig$values) %*% t(eig$vectors)
  }
  Sigma_sh[t,,] <- S
}

cat(sprintf("shrinkTVPVAR: extracción completada. Te = %d\n", Te_sh))

# ------------------------------------------------------------------
# GFEVD y Conectividad
# ------------------------------------------------------------------
gfevd_sh <- array(0, dim = c(Te_sh, k, k))
for (t in 1:Te_sh)
  gfevd_sh[t,,] <- compute_gfevd(A1_sh[t,,], A2_sh[t,,], Sigma_sh[t,,], H, k)

conn_sh <- compute_connectedness(gfevd_sh, k, Te_sh, var_names)

# -- Gráfico TCI --
dates_sh <- seq(as.Date("2020-01-03"), by = "days", length.out = Te_sh)
df_tci_sh <- data.frame(Date = dates_sh, TCI = conn_sh$TCI)
p_tci_sh <- ggplot(df_tci_sh, aes(x = Date, y = TCI)) +
  geom_area(fill = "#E29A4A", alpha = 0.3) +
  geom_line(color = "#663300", linewidth = 1) +
  theme_minimal(base_size = 14) +
  labs(title = "TCI – Estimación Bayesiana (shrinkTVPVAR)", x = "Fecha", y = "TCI (%)") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid.minor = element_blank())
print(p_tci_sh)

# -- Gráfico NET --
df_net_sh <- as.data.frame(conn_sh$NET)
df_net_sh$Date <- dates_sh
df_net_sh_long <- pivot_longer(df_net_sh, all_of(var_names),
                               names_to = "Variable", values_to = "NetValue") %>%
  mutate(Role = ifelse(NetValue > 0, "Transmisor Neto", "Receptor Neto"))

p_net_sh <- ggplot(df_net_sh_long, aes(x = Date, y = NetValue, fill = Role)) +
  geom_col(width = 1.2, alpha = 0.85, position = "identity") +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  facet_wrap(~ Variable, scales = "fixed", ncol = 2) +
  scale_fill_manual(values = c("Transmisor Neto" = "#D35400",
                               "Receptor Neto"   = "#7F8C8D")) +
  theme_minimal(base_size = 14) +
  labs(title = "Conectividad Neta por Variable – shrinkTVPVAR",
       x = "Fecha", y = "Conectividad Neta (%)", fill = "Rol:") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid.minor = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face = "bold", size = 12),
        strip.background = element_rect(fill = "gray90", color = NA))
print(p_net_sh)

# -- Gráfico de red --
plot_net_graph(conn_sh$NET, conn_sh$NPDC, var_names,
               "Red de Conectividad Neta (Estimación Bayesiana shrinkTVPVAR)")   



#GRAFICAS CON DeepTVAR

library(reticulate)
library(ggplot2)
library(tidyr)
library(dplyr)

np <- import("numpy")

# Cargar las estimaciones de DeepTVAR
A1_estimates <- np$load("A1_mean_sim (1).npy")
A2_estimates <- np$load("A2_mean_sim (1).npy")
Sigma_estimates <- np$load("Sigma_mean_sim (1).npy")

# Verificar las dimensiones (T, k, k)
print(dim(A1_estimates))
print(dim(Sigma_estimates))

#  Configurar dimensiones automáticamente basándose en las matrices cargadas
T_deep <- dim(A1_estimates)[1] # Extrae T automáticamente (debería ser 498)
k <- dim(A1_estimates)[2]      # Extrae k automáticamente (debería ser 4)
p <- 2
H <- 12 # Horizonte de predicción

# Si var_names no está definido, lo definimos aquí
var_names <- paste0("V", 1:k) 

# Iniciar un array vacío para guardar la GFEVD para cada tiempo t
gfevd_deep <- array(0, dim = c(T_deep, k, k))

# Bucle usando las matrices de DeepTVAR
for (t in 1:T_deep) {
  gfevd_deep[t,,] <- compute_gfevd(A1_estimates[t,,], A2_estimates[t,,], Sigma_estimates[t,,], H, k)
}

# Calcular conectividad
conn_deep <- compute_connectedness(gfevd_deep, k, T_deep, var_names)


# GRÁFICOS DeepTVAR

# Fechas ajustadas al número de observaciones de DeepTVAR
dates_deep <- seq(as.Date("2020-01-03"), by = "days", length.out = T_deep)

# -- Gráfico TCI --
df_tci_deep <- data.frame(Date = dates_deep, TCI = conn_deep$TCI)

p_tci_deep <- ggplot(df_tci_deep, aes(x = Date, y = TCI)) +
  geom_area(fill = "#9B59B6", alpha = 0.3) +   # Color diferente para DeepTVAR
  geom_line(color = "#4A235A", linewidth = 1) +
  theme_minimal(base_size = 14) +
  labs(title = "TCI – Estimación DeepTVAR", x = "Fecha", y = "TCI (%)") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid.minor = element_blank())
print(p_tci_deep)

# -- Gráfico NET --
df_net_deep <- as.data.frame(conn_deep$NET)
df_net_deep$Date <- dates_deep
df_net_deep_long <- pivot_longer(df_net_deep, all_of(var_names),
                                 names_to = "Variable", values_to = "NetValue") %>%
  mutate(Role = ifelse(NetValue > 0, "Transmisor Neto", "Receptor Neto"))

p_net_deep <- ggplot(df_net_deep_long, aes(x = Date, y = NetValue, fill = Role)) +
  geom_col(width = 1.2, alpha = 0.85, position = "identity") +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  facet_wrap(~ Variable, scales = "fixed", ncol = 2) +
  scale_fill_manual(values = c("Transmisor Neto" = "#D35400",
                               "Receptor Neto"   = "#7F8C8D")) +
  theme_minimal(base_size = 14) +
  labs(title = "Conectividad Neta por Variable – DeepTVAR",
       x = "Fecha", y = "Conectividad Neta (%)", fill = "Rol:") +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        panel.grid.minor = element_blank(),
        legend.position = "bottom",
        strip.text = element_text(face = "bold", size = 12),
        strip.background = element_rect(fill = "gray90", color = NA))
print(p_net_deep)

# -- Gráfico de red --
plot_net_graph(conn_deep$NET, conn_deep$NPDC, var_names,
               "Red de Conectividad Neta (Estimación DeepTVAR)")
