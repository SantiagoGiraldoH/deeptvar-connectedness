import optuna
import torch
import random
import numpy as np
import pandas as pd
import torch.optim as optim


# CONFIGURACIÓN GLOBAL

max_epochs = 2000
K_vars = 4               
p_lags = 2
tol = 1e-4             
patience = 50         

num_replications_tuning = 1 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Ejecutando el estudio en: {device}")


# DEFINIR LA FUNCIÓN OBJETIVO PARA OPTUNA

def objective(trial):
    # Semillas para reproducibilidad
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Espacio de Búsqueda
    hidden_size = trial.suggest_categorical("hidden_size", [16, 24, 32, 48])
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.6)
    trial_val_losses = []

    # Bucle (Si es 1 sola serie real, este bucle corre 1 vez)
    for i in range(1, num_replications_tuning + 1):
        file_path = '/kaggle/input/datasets/santiagogiraldohenao/series-equipos/series_equipos_modelo.csv'
        
        try:
            df_y = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"No se encontró: {file_path}")
            break
            
        y_data = df_y.iloc[:, 1:].values 
        T_total = len(y_data)
        
        # Split de Entrenamiento (80%) y Validación (20%)
        split_idx = int(T_total * 0.8)
        
        # Se estandariza usando solo la media y std del set de entrenamiento
        y_mean = np.mean(y_data[:split_idx], axis=0)
        y_std = np.std(y_data[:split_idx], axis=0)
        y_data_scaled = (y_data - y_mean) / y_std
        
        y_tensor = torch.tensor(y_data_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Generar z_t dinámicamente según el tamaño de la serie real
        z_input = create_time_features(T=T_total).to(device)
        
        # Inicializar el modelo
        model = DeepTVAR(k=K_vars, p=p_lags, hidden_size=hidden_size, dropout_rate=dropout_rate).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val_loss = float('inf')
        best_loss = float('inf')
        patience_counter = 0
        
        # Bucle de Entrenamiento
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            
            A1_pred, A2_pred, Sigma_pred, L_pred = model(z_input)
            
            # Recortar tensores hasta split_idx para el Loss de Entrenamiento
            y_train = y_tensor[:, :split_idx, :]
            A1_train = A1_pred[:, :split_idx, :, :]
            A2_train = A2_pred[:, :split_idx, :, :]
            L_train = L_pred[:, :split_idx, :, :]
            
            # Calcular Loss en el set de entrenamiento
            loss = gaussian_log_likelihood_loss(y_train, A1_train, A2_train, L_train, p=p_lags)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Control de Convergencia (sobre el loss de entrenamiento)
            current_loss = loss.item()
            rel_change = abs(best_loss - current_loss) / (abs(best_loss) + 1e-8)

            if current_loss < best_loss:
                best_loss = current_loss
            
            if rel_change < tol:
                patience_counter += 1  
            else:
                patience_counter = 0   
            
            if patience_counter >= patience:
                break
                
        #Evaluación
        model.eval()
        with torch.no_grad():
            A1_final, A2_final, Sigma_final, L_final = model(z_input)
            
            # Recortar tensores para el periodo de validación. 
            # Retrocedemos 'p_lags' para que el VAR tenga los datos históricos (t-1, t-2) necesarios para predecir el primer punto de validación.
            val_start_idx = split_idx - p_lags
            
            y_val = y_tensor[:, val_start_idx:, :]
            A1_val = A1_final[:, val_start_idx:, :, :]
            A2_val = A2_final[:, val_start_idx:, :, :]
            L_val = L_final[:, val_start_idx:, :, :]
            
            # Calcular el Log-Likelihood fuera de muestra
            val_loss = gaussian_log_likelihood_loss(y_val, A1_val, A2_val, L_val, p=p_lags).item()
            trial_val_losses.append(val_loss)
            
            print(f"Épocas: {epoch+1} | NLL Train: {best_loss:.4f} | NLL Validación (Out-of-Sample): {val_loss:.4f}")
            
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience: 
                break


    # Optuna minimiza la pérdida de VALIDACIÓN
    avg_val_loss = np.mean(trial_val_losses)
    
    print(f"--> Trial Finalizado | H_Size: {hidden_size}, LR: {lr:.5f}, WD: {weight_decay:.5f} | Val Loss: {avg_val_loss:.6f}")
    
    return avg_val_loss


# INICIAR EL ESTUDIO DE OPTUNA

print("Iniciando Búsqueda de Hiperparámetros Empíricos con Optuna...")
study = optuna.create_study(direction="minimize", study_name="DeepTVAR_Empirical")

study.optimize(
    objective, 
    n_trials= 15,
    timeout=20000,
    callbacks=[save_checkpoint_callback] 
)

print("\n=== MEJORES HIPERPARÁMETROS (DATOS REALES) ===")
print(study.best_params)
print(f"Mejor NLL de Validación: {study.best_value:.6f}")
