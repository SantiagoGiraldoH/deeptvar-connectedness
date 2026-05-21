# Simulacion TV-VAR(2) model
set.seed(42)
n_obs <- 500 
n_dims <- 4


Y <- matrix(0, nrow = n_obs, ncol = n_dims)
Sigma_u <- diag(0.1, n_dims) # Error covariance
errors <- mvtnorm::rmvnorm(n_obs, sigma = Sigma_u)

intercept_t <- rep(0, n_dims)
phi1_t <- diag(0.4, n_dims)
phi2_t <- diag(0.2, n_dims)

# (Random Walk)
params <- c(intercept_t, as.vector(phi1_t), as.vector(phi2_t))
n_params <- length(params)

true_params_history <- matrix(0, nrow = n_obs, ncol = n_params)

for (t in 1:n_obs) {
  
  params <- params + rnorm(n_params, mean = 0, sd = 0.01)
  true_params_history[t, ] <- params
  
  
  c_t <- params[1:n_dims]
  Phi1 <- matrix(params[(n_dims + 1):(n_dims + n_dims^2)], n_dims, n_dims)
  Phi2 <- matrix(params[(n_dims + n_dims^2 + 1):n_params], n_dims, n_dims)
  
  
  if (t <= 2) {
    Y[t, ] <- errors[t, ]
  } else {
    Y[t, ] <- c_t + Phi1 %*% Y[t-1, ] + Phi2 %*% Y[t-2, ] + errors[t, ]
  }
}


plot.ts(Y, main = "Simulado 4D TV-VAR(2) Proceso")


# EXPORTACIÓN COMPATIBLE CON PYTHON

#Exportar A1 y A2 
A1_true <- true_params_history[, (n_dims + 1):(n_dims + n_dims^2)]
A1_python_compatible <- t(apply(A1_true, 1, function(row) {
  mat <- matrix(row, n_dims, n_dims, byrow = FALSE) 
  as.vector(t(mat)) 
}))
write.csv(A1_python_compatible, "A1_true.csv", row.names = FALSE)

A2_true <- true_params_history[, (n_dims + n_dims^2 + 1):n_params]
A2_python_compatible <- t(apply(A2_true, 1, function(row) {
  mat <- matrix(row, n_dims, n_dims, byrow = FALSE)
  as.vector(t(mat))
}))
write.csv(A2_python_compatible, "A2_true.csv", row.names = FALSE)


get_lower_tri_python_style <- function(mat) {
  vals <- c()
  for (i in 1:nrow(mat)) {
    for (j in 1:i) {
      vals <- c(vals, mat[i, j])
    }
  }
  return(vals)
}

Sigma_history_10 <- matrix(0, nrow = n_obs, ncol = 10)
for(t in 1:n_obs) {
  
  Sigma_history_10[t, ] <- get_lower_tri_python_style(Sigma_u) 
}
write.csv(Sigma_history_10, "Sigma_true.csv", row.names = FALSE)

# Exportar serie simulada
df_export <- data.frame(Index = 1:n_obs, Y)
colnames(df_export) <- c("Index", paste0("V", 1:n_dims))
write.csv(df_export, file = "serie_simulada_con_indice.csv", row.names = FALSE)
