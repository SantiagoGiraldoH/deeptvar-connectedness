def gaussian_log_likelihood_loss(y, A1, A2, L, p=2):
    """
    Calcula el negativo de la log-verosimilitud condicional para un VAR(2).
    y: Tensor de datos, forma (batch_size, seq_len, k) 
    A1, A2: Coeficientes causales, forma (batch_size, seq_len, k, k)
    L: Factor de Cholesky de Sigma_t, forma (batch_size, seq_len, k, k)
    """
    batch_size, seq_len, k = y.size()
    loss = 0.0
    
    # Iteramos desde el rezago p hasta el final de la serie
    for t in range(p, seq_len):
        # Extraemos las observaciones y las ajustamos para multiplicación matricial (agregamos dimensión extra)
        # Forma resultante: (batch_size, 4, 1)
        y_t = y[:, t, :].unsqueeze(2)     
        y_t_1 = y[:, t-1, :].unsqueeze(2) 
        y_t_2 = y[:, t-2, :].unsqueeze(2) 
        
        # Extraemos las matrices en el instante t
        A1_t = A1[:, t, :, :] # (batch_size, 4, 4)
        A2_t = A2[:, t, :, :] # (batch_size, 4, 4)
        L_t = L[:, t, :, :]   # (batch_size, 4, 4)
        
        # Calcular el error de predicción (epsilon_t)
        # torch.bmm realiza la multiplicación de matrices en batch
        pred_t = torch.bmm(A1_t, y_t_1) + torch.bmm(A2_t, y_t_2)
        epsilon_t = y_t - pred_t # (batch_size, 4, 1)
        
        # Calcular log |Sigma_t|
        # Extraemos la diagonal de L_t y calculamos la suma de logaritmos
        diag_L = torch.diagonal(L_t, dim1=-2, dim2=-1)
        # Usamos abs() y sumamos 1e-6 para evitar log(negativos) y log(0)
        log_det_Sigma = 2.0 * torch.sum(torch.log(torch.abs(diag_L) + 1e-6), dim=-1)
        
        # Calcular la forma cuadrática: epsilon_t' * Sigma_t^{-1} * epsilon_t
        # Usamos solve_triangular para máxima estabilidad (upper=False porque L_t es triangular inferior)
        v = torch.linalg.solve_triangular(L_t, epsilon_t, upper=False)
        quadratic_term = torch.sum(v ** 2, dim=-2).squeeze(-1) # (batch_size,)
        
        # Sumar a la pérdida total temporal
        # Omitimos la constante m*log(2*pi) porque no afecta el cálculo del gradiente
        loss += 0.5 * (log_det_Sigma + quadratic_term)
        
    # Retornamos el promedio sobre el batch y normalizamos por el número de instantes efectivos
    return torch.mean(loss) / (seq_len - p)
