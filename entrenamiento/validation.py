# CÁLCULO DE MÉTRICAS


model.load_state_dict(torch.load('best_model_weights.pt'))
model.eval()

with torch.no_grad():
    A1_final, A2_final, Sigma_final, _ = model(z_input)
    
    A1_est_np = A1_final.squeeze(0).cpu().numpy()
    A2_est_np = A2_final.squeeze(0).cpu().numpy()
    Sigma_est_np = Sigma_final.squeeze(0).cpu().numpy()
    
    val_start_idx = split_idx - p_lags
    y_val_real_scaled = y_data_scaled[val_start_idx:]
    A1_val_est = A1_est_np[val_start_idx:]
    A2_val_est = A2_est_np[val_start_idx:]

    T_val = len(y_val_real_scaled)
    y_pred_scaled = np.zeros((T_val, K_vars))

    for t in range(p_lags, T_val):
        y_lag1 = y_val_real_scaled[t-1] 
        y_lag2 = y_val_real_scaled[t-2] 
        
        term1 = np.dot(A1_val_est[t], y_lag1)
        term2 = np.dot(A2_val_est[t], y_lag2)
        y_pred_scaled[t] = term1 + term2
    # Recortar los rezagos iniciales para comparar limpiamente
    y_val_real_eval_scaled = y_val_real_scaled[p_lags:]
    y_pred_eval_scaled = y_pred_scaled[p_lags:]

    # DESESTANDARIZACIÓN PARA MÉTRICAS REALES
    # Transformación inversa: Original = Scaled * Std + Mean
    y_val_real_eval = (y_val_real_eval_scaled * (y_std + 1e-8)) + y_mean
    y_pred_eval = (y_pred_eval_scaled * (y_std + 1e-8)) + y_mean
    
    
    forecast_errors = y_val_real_eval - y_pred_eval

var_names = ["Equipo_1", "Equipo_2", "Equipo_3", "Equipo_4"]
metricas_evaluacion = {"por_variable": {}, "global": {}}

print("\n=== RESULTADOS OUT-OF-SAMPLE ===")
for k in range(K_vars):
    errors_k = forecast_errors[:, k]
    real_k   = y_val_real_eval[:, k]

    rmse_k = float(np.sqrt(np.mean(errors_k**2)))
    mae_k  = float(np.mean(np.abs(errors_k)))
    mse_k  = float(np.mean(errors_k**2))
    
    # WAPE por variable en la escala original
    wape_k = float((np.sum(np.abs(errors_k)) / (np.sum(np.abs(real_k)) + 1e-8)) * 100)
    
    print(f"Variable: {var_names[k]}")
    print(f"  MSE:   {mse_k:.4f}")
    print(f"  RMSE:  {rmse_k:.4f}")
    print(f"  MAE:   {mae_k:.4f}")
    print(f"  WAPE:  {wape_k:.2f}%")
    
    metricas_evaluacion["por_variable"][var_names[k]] = {
        "MSE": mse_k, "RMSE": rmse_k, "MAE": mae_k, "WAPE": wape_k
    }

global_mse  = float(np.mean(forecast_errors**2))
global_rmse = float(np.sqrt(global_mse))
global_mae  = float(np.mean(np.abs(forecast_errors)))

# WAPE global matricial en la escala original
global_wape = float((np.sum(np.abs(forecast_errors)) / (np.sum(np.abs(y_val_real_eval)) + 1e-8)) * 100)

print("\n--- DESEMPEÑO GLOBAL DEL SISTEMA ---")
print(f"Global MSE:   {global_mse:.4f}")
print(f"Global RMSE:  {global_rmse:.4f}")
print(f"Global MAE:   {global_mae:.4f}")
print(f"Global WAPE:  {global_wape:.2f}%")
print(f"Global NLL:   {best_val_loss:.4f}")

metricas_evaluacion["global"] = {
    "MSE": global_mse,
    "RMSE": global_rmse,
    "MAE": global_mae,
    "WAPE": global_wape,
    "NLL": float(best_val_loss)
}
