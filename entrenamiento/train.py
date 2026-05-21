import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Reproducibilidad
torch.manual_seed(42)
np.random.seed(42)


hidden_size = 32         
learning_rate = 2.4e-5   
weight_decay = 0.0027    
dropout_rate = 0.4

max_epochs = 3000        
K_vars = 4               
p_lags = 2
tol = 1e-4             
patience = 50          

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Ejecutando evaluación Out-of-Sample en: {device}")


file_path = '/kaggle/input/datasets/santiagogiraldohenao/series-equipos/series_equipos_modelo.csv'
try:
    df_y = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo: {file_path}")
    exit()

y_data = df_y.iloc[:, 1:].values 
T_total = len(y_data)

# Utilizar si se quiere Partición 80% Entrenamiento / 20% Prueba
split_idx = T_total

# Estandarización SÓLO con la media y std del entrenamiento (Evita Data Leakage)
y_mean = np.mean(y_data[:split_idx], axis=0)
y_std = np.std(y_data[:split_idx], axis=0)
y_data_scaled = (y_data - y_mean) / (y_std + 1e-8)

y_tensor = torch.tensor(y_data_scaled, dtype=torch.float32).unsqueeze(0).to(device)
z_input = create_time_features(T=T_total).to(device)

model = DeepTVAR(k=K_vars, p=p_lags, hidden_size=hidden_size, dropout_rate=dropout_rate).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


best_loss = float('inf')
patience_counter = 0

print(f"Iniciando entrenamiento en el {split_idx} primeros periodos...")
loss_history = []
for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()
    
    A1_pred, A2_pred, Sigma_pred, L_pred = model(z_input)
    
    # Recortamos todo hasta split_idx para que no vea el futuro
    y_train = y_tensor[:, :split_idx, :]
    A1_train = A1_pred[:, :split_idx, :, :]
    A2_train = A2_pred[:, :split_idx, :, :]
    L_train = L_pred[:, :split_idx, :, :]
    
    loss = gaussian_log_likelihood_loss(y_train, A1_train, A2_train, L_train, p=p_lags)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    current_loss = loss.item()
    loss_history.append(current_loss)
   
    
    rel_change = abs(best_loss - current_loss) / (abs(best_loss) + 1e-8)

    if current_loss < best_loss:
        best_loss = current_loss
    
    if rel_change < tol:
        patience_counter += 1  
    else:
        patience_counter = 0   
    
    if patience_counter >= patience:
        print(f"Convergencia alcanzada en la época {epoch}. Loss Train: {best_loss:.4f}")
        break
    if (epoch + 1) % 100 == 0:
        print(f"Época {epoch+1:04d}/{max_epochs} | NLL Train Completo: {current_loss:.4f}")
