library(tvReg)
library(mvtnorm)


data <- read.csv("series_equipos_modelo.csv")

Y <- as.matrix(data[, -1])

T_total <- 1055
T_train <- floor(0.8 * T_total) # ~844 observaciones
T_test <- T_total - T_train     # ~211 observaciones


y_mean <- colMeans(Y[1:T_train, ])


calc_np_std <- function(x) sqrt(sum((x - mean(x))^2) / length(x))
y_std <- apply(Y[1:T_train, ], 2, calc_np_std)

Y <- sweep(Y, 2, y_mean, "-")
Y <- sweep(Y, 2, y_std, "/")
p <- 2 
k <- ncol(Y)

# Separar datos de entrenamiento
Y_train <- Y[1:T_train, ]
Y_test <- Y[(T_train + 1):T_total, ]
Y_test_orig <- sweep(sweep(Y_test, 2, y_std, "*"), 2, y_mean, "+")

bw_opt <- tvVAR(Y_train, p = p, type = "const", est = "ll")$bw 

# Matrices para almacenar resultados
Y_pred_tvreg <- matrix(NA, nrow = T_test, ncol = k)
colnames(Y_pred_tvreg) <- colnames(Y)

# Vector para guardar el NLL de cada paso
nll_tvreg_vals <- numeric(T_test) 



for (t in 1:T_test) {
  
  
  Y_current <- Y[1:(T_train + t - 1), ]
  T_current <- nrow(Y_current)
  
  # Ajustar el modelo con el bw fijo 
  model_tv <- tvVAR(Y_current, p = p, type = "const", bw = bw_opt, est = "ll") 

  
  tvcoef_obj <- model_tv$coefficients  
  num_coefs <- (p * k) + 1 
  coef_T <- matrix(NA, nrow = k, ncol = num_coefs)
  
  
  if (is.array(tvcoef_obj) && length(dim(tvcoef_obj)) == 3) {
    ultimos_coefs <- tvcoef_obj[dim(tvcoef_obj)[1], , ]
    
    if (is.matrix(ultimos_coefs)) {
      if (nrow(ultimos_coefs) == num_coefs && ncol(ultimos_coefs) == k) {
        coef_T <- t(ultimos_coefs)
      } else {
        coef_T <- ultimos_coefs
      }
    } else {
      coef_T[1, ] <- as.numeric(ultimos_coefs)
    }
  } 
  
  else if (is.list(tvcoef_obj)) {
    for (i in 1:k) {
      matriz_eq <- tvcoef_obj[[i]]
      coef_T[i, ] <- as.numeric(matriz_eq[nrow(matriz_eq), ])
    }
  } else {
    stop()
  }
  
  
  y_lag1 <- as.numeric(Y_current[T_current, ])
  y_lag2 <- as.numeric(Y_current[T_current - 1, ])
  
  
  regresores <- c(1, y_lag1, y_lag2)
  
  
  pred_1_step <- as.numeric(coef_T %*% regresores)
  Y_pred_tvreg[t, ] <- pred_1_step
  

  # Calcular el NLL en este paso (Log-Likelihood)
  
  
  sigma_u <- cov(residuals(model_tv))
  y_real <- as.numeric(Y[T_train + t, ])
  
  
  ll_t <- dmvnorm(x = y_real, mean = pred_1_step, sigma = sigma_u, log = TRUE)
  nll_tvreg_vals[t] <- -ll_t
  
  if (t %% 20 == 0) cat(sprintf("Paso %d de %d completado...\n", t, T_test))
}

# Calcular Métricas Finales
mae_tvreg    <- colMeans(abs(Y_test - Y_pred_tvreg))
mse_tvreg    <- colMeans((Y_test - Y_pred_tvreg)^2)
rmse_tvreg   <- sqrt(colMeans((Y_test - Y_pred_tvreg)^2))
nll_global   <- mean(nll_tvreg_vals)
mse_global   <- mean((Y_test - Y_pred_tvreg)^2)
mae_global   <- mean(abs(Y_test - Y_pred_tvreg))
rmse_global  <- sqrt(mse_global)


Y_pred_orig_tv      <- sweep(sweep(Y_pred_tvreg, 2, y_std, "*"), 2, y_mean, "+")
wape_global_tvreg   <- sum(abs(Y_test_orig - Y_pred_orig_tv)) /
  sum(abs(Y_test_orig)) * 100
mse_global
mae_global  
rmse_global
nll_global
wape_global_tvreg
# Mostrar resultados

cat("   RESULTADOS OUT-OF-SAMPLE: \n")
cat("NLL Global en Validación:", round(nll_global, 4), "\n\n")

print(data.frame(
  Variable = colnames(Y),
  MAE = mae_tvreg,
  RMSE = rmse_tvreg
))

###############################################################################
# Espacio-Estado - shrinkTVPVAR

library(shrinkTVPVAR)
library(mvtnorm)


# Matrices de resultados

Y_pred_shrink   <- matrix(NA, nrow = T_test, ncol = k)
colnames(Y_pred_shrink) <- colnames(Y)
nll_shrink_vals     <- numeric(T_test)   
nll_shrink_vals_noc <- numeric(T_test)   


log2pi_const <- 0.5 * k * log(2 * pi)


set.seed(123)


for (t in 1:T_test) {
  
  Y_current <- Y[1:(T_train + t - 1), ]
  
  # Ajuste TVP-VAR bayesiano
  model_tv <- shrinkTVPVAR(Y_current, p = p, niter = 1000, nburn = 500,
                           display_progress = FALSE)
  

  # Predicción puntual: media a posteriori de la distribución predictiva

  pred_obj    <- forecast_shrinkTVPVAR(model_tv, n.ahead = 1)
  pred_matrix <- sapply(1:k, function(i) as.numeric(pred_obj[[i]]$y_pred))
  pred_1_step <- colMeans(pred_matrix)
  Y_pred_shrink[t, ] <- pred_1_step
  

  # COVARIANZA CONDICIONAL Sigma_t
  
  Sigma_array  <- model_tv$Sigma                      
  T_eff        <- dim(Sigma_array)[3]                  
  Sigma_last   <- Sigma_array[, , T_eff, ]             
  

  sigma_u <- apply(Sigma_last, c(1, 2), mean)          # [k, k]
  

  sigma_u <- 0.5 * (sigma_u + t(sigma_u))
  
 
  # NLL del paso t

  y_real <- as.numeric(Y[T_train + t, ])
  ll_t   <- dmvnorm(x = y_real, mean = pred_1_step, sigma = sigma_u, log = TRUE)
  
  nll_shrink_vals[t]     <- -ll_t                          
  nll_shrink_vals_noc[t] <- -ll_t - log2pi_const           
  
  if (t %% 10 == 0) cat(sprintf("Paso %d de %d completado...\n", t, T_test))
}

# Métricas finales

mse_global             <- mean((Y_test - Y_pred_shrink)^2)
mae_global             <- mean(abs(Y_test - Y_pred_shrink))
rmse_global            <- sqrt(mse_global)
nll_global_shrink      <- mean(nll_shrink_vals)
nll_global_shrink_noc  <- mean(nll_shrink_vals_noc)
mae_shrink             <- colMeans(abs(Y_test - Y_pred_shrink))
rmse_shrink            <- sqrt(colMeans((Y_test - Y_pred_shrink)^2))

# WAPE en escala original
Y_pred_orig_shrink <- sweep(sweep(Y_pred_shrink, 2, y_std, "*"), 2, y_mean, "+")
wape_global_shrink <- sum(abs(Y_test_orig - Y_pred_orig_shrink)) /
  sum(abs(Y_test_orig)) * 100


cat("   RESULTADOS OUT-OF-SAMPLE (shrinkTVPVAR): \n")

cat("MSE  Global:", round(mse_global, 4), "\n")
cat("MAE  Global:", round(mae_global, 4), "\n")
cat("RMSE Global:", round(rmse_global, 4), "\n")
cat("WAPE Global:", round(wape_global_shrink, 4), "\n")
cat("NLL  (con constante, ) :",
    round(nll_global_shrink, 4), "\n")
cat("NLL  (sin constante,)  :",
    round(nll_global_shrink_noc, 4), "\n\n")

print(data.frame(
  Variable = colnames(Y),
  MAE  = mae_shrink,
  RMSE = rmse_shrink
))


###############################################################################
library(vars)
library(forecast)
library(mvtnorm)


Y_pred_var <- matrix(NA, nrow = T_test, ncol = k)
colnames(Y_pred_var) <- colnames(Y)

# Vector para almacenar el NLL en cada paso
nll_var_vals <- numeric(T_test) 

for (t in 1:T_test) {
  Y_current <- Y[1:(T_train + t - 1), ]
  
  # Ajustar VAR estándar (MCO)
  model_var <- VAR(Y_current, p = p, type = "const")
  
  # Pronóstico a 1 paso adelante
  fcst_var <- predict(model_var, n.ahead = 1)
  
  # Vector temporal para guardar la predicción de este instante
  pred_1_step <- numeric(k)
  
  # Extraer la predicción puntual de cada variable
  for(i in 1:k) {
    pred_1_step[i] <- fcst_var$fcst[[i]][1, "fcst"]
    Y_pred_var[t, i] <- pred_1_step[i]
  }
  
  #  CÁLCULO DEL NLL EN ESTE PASO

  # Matriz de covarianza de los residuos del VAR estimado
  sigma_u <- cov(resid(model_var))
  
  y_real <- as.numeric(Y_test[t, ]) 

  ll_t <- dmvnorm(x = y_real, mean = pred_1_step, sigma = sigma_u, log = TRUE)
  nll_var_vals[t] <- -ll_t
  # -------------------------------------------------------------
  
  if (t %% 20 == 0) cat(sprintf("Paso %d de %d completado...\n", t, T_test))
}

# Calcular métricas finales
mae_var        <- colMeans(abs(Y_test - Y_pred_var))
rmse_var       <- sqrt(colMeans((Y_test - Y_pred_var)^2))
nll_global_var <- mean(nll_var_vals)
mse_global_var <- mean((Y_test - Y_pred_var)^2)
mae_global_var <- mean(abs(Y_test - Y_pred_var))
rmse_global_var <- sqrt(mse_global_var)


Y_pred_orig_var    <- sweep(sweep(Y_pred_var, 2, y_std, "*"), 2, y_mean, "+")
wape_global_var    <- sum(abs(Y_test_orig - Y_pred_orig_var)) /
  sum(abs(Y_test_orig)) * 100
mse_global_var
mae_global_var  
rmse_global
nll_global_var
wape_global_var

cat("   RESULTADOS OUT-OF-SAMPLE: VAR CLÁSICO (Ventana Expansiva)  \n")

cat("NLL Global:", round(nll_global_var, 4), "\n\n")

print(data.frame(
  Variable = colnames(Y),
  MAE = mae_var,
  RMSE = rmse_var
))

