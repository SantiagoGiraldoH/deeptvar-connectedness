import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Reproducibilidad
torch.manual_seed(42)
np.random.seed(42)


# HIPERPARÁMETROS ÓPTIMOS

hidden_size = 32         
learning_rate = 2.4e-5   
weight_decay = 0.0027    
dropout_rate = 0.55      

max_epochs = 3000        
K_vars = 4               
p_lags = 2
tol = 1e-4             
patience = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Ejecutando evaluación Out-of-Sample en: {device}")


#  CARGA Y PARTICIÓN DE DATOS

file_path = '/kaggle/input/datasets/santiagogiraldohenao/series-equipos/series_equipos_modelo.csv'
try:
    df_y = pd.read_csv(file_path)
except FileNotFoundError:
    
    print(f"Error: No se encontró el archivo: {file_path}. Usando ruta local...")
    df_y = pd.read_csv('series_equipos_modelo.csv')

y_data = df_y.iloc[:, 1:].values 
T_total = len(y_data)

# Partición 80% Entrenamiento / 20% Validación
split_idx = int(T_total * 0.8)

# Estandarización SÓLO con la media y std del entrenamiento (Evita Data Leakage)
y_mean = np.mean(y_data[:split_idx], axis=0)
y_std = np.std(y_data[:split_idx], axis=0)
# Sumamos un pequeño epsilon para evitar división por cero
y_data_scaled = (y_data - y_mean) / (y_std + 1e-8)

y_tensor = torch.tensor(y_data_scaled, dtype=torch.float32).unsqueeze(0).to(device)
z_input = create_time_features(T=T_total).to(device)

# Inicializar modelo (

model = DeepTVAR(k=K_vars, p=p_lags, hidden_size=hidden_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# BUCLE DE ENTRENAMIENTO (CON VALIDACIÓN)

best_val_loss = float('inf')
train_loss_history = []
val_loss_history = []

print(f"\nIniciando entrenamiento...")
print(f"Observaciones Train: {split_idx}")
print(f"Observaciones Val: {T_total - split_idx}\n")

for epoch in range(max_epochs):
    #FASE DE ENTRENAMIENTO
    model.train()
    optimizer.zero_grad()
    
    A1_pred, A2_pred, Sigma_pred, L_pred = model(z_input)
    
    # Recortamos para el conjunto de Entrenamiento
    y_train = y_tensor[:, :split_idx, :]
    A1_train = A1_pred[:, :split_idx, :, :]
    A2_train = A2_pred[:, :split_idx, :, :]
    L_train = L_pred[:, :split_idx, :, :]
    
    # Loss de entrenamiento
    train_loss = gaussian_log_likelihood_loss(y_train, A1_train, A2_train, L_train, p=p_lags)
    train_loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    #FASE DE VALIDACIÓN
    model.eval()
    with torch.no_grad():
        # Recortamos para el conjunto de Validación (desde split_idx - p_lags)
        val_start = split_idx - p_lags
        y_val = y_tensor[:, val_start:, :]
        A1_val = A1_pred[:, val_start:, :, :]
        A2_val = A2_pred[:, val_start:, :, :]
        L_val = L_pred[:, val_start:, :, :]
        
        val_loss = gaussian_log_likelihood_loss(y_val, A1_val, A2_val, L_val, p=p_lags)
    
    # Guardar métricas
    current_train_loss = train_loss.item()
    current_val_loss = val_loss.item()
    
    train_loss_history.append(current_train_loss)
    val_loss_history.append(current_val_loss)
    
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        # Guardar los mejores pesos basados en validación
        torch.save(model.state_dict(), 'best_model_weights.pt')
    
    # Imprimir progreso
    if (epoch + 1) % 100 == 0: 
        print(f"Época {epoch+1:04d}/{max_epochs} | Train NLL: {current_train_loss:.4f} | Val NLL: {current_val_loss:.4f} | Mejor Val: {best_val_loss:.4f}")

print("\nEntrenamiento finalizado.")
print(f"El mejor NLL de Validación (Out-of-Sample) fue: {best_val_loss:.4f}")


# GRÁFICA DE CONVERGENCIA

plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Entrenamiento (Train NLL)', color='blue', linewidth=2)
plt.plot(val_loss_history, label='Validación (Val NLL)', color='#ff7f0e', linewidth=2)

plt.title('Curva de Aprendizaje: Evolución de la Log-Verosimilitud Negativa (NLL)', fontsize=14)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Negative Log-Likelihood (NLL)', fontsize=12)


best_epoch = np.argmin(val_loss_history)
plt.scatter(best_epoch, best_val_loss, color='gray', zorder=5, label=f'Mejor Época ({best_epoch})')

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('curva_aprendizaje_deeptvar.png', dpi=300)
plt.show()
