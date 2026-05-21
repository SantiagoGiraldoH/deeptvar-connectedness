class DeepTVAR(nn.Module):
    def __init__(self, k, p, hidden_size, num_layers=1, dropout_rate=0.0):
        super(DeepTVAR, self).__init__()
        self.k = k # k = 4 variables 
        self.p = p # p = 2 rezagos
        self.hidden_size = hidden_size
        
        # El input_size z_t = (t, t^2, t^3, 1/t, 1/t^2, 1/t^3)
        self.input_size = 6

        # Capa LSTM
        # batch_first=True tensores con forma (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.num_A_elements = self.p * (self.k ** 2)
        self.num_L_elements = int(self.k * (self.k + 1) / 2)
        self.total_output_dim = self.num_A_elements + self.num_L_elements # 42
        
        # Capa lineal que representa W_a * h_t + b_a
        self.linear = nn.Linear(self.hidden_size, self.total_output_dim)
    
       
    def forward(self, z):
        batch_size, seq_len, _ = z.size()
        
        # z es el tensor de tiempo con forma (1, T, 6)
        # lstm_out contiene el estado oculto h_t para cada instante t
        lstm_out, _ = self.lstm(z)
        lstm_out = self.dropout(lstm_out)
        a_t = self.linear(lstm_out)# Forma: (1, 500, 42)

        # Separar a_t en coeficientes preliminares y elementos de Cholesky
        # A_elements forma: (1, 500, 32)
        A_elements = a_t[:, :, :self.num_A_elements] 
        # L_elements forma: (1, 500, 10)
        L_elements = a_t[:, :, self.num_A_elements:]

        # Reconstruir las matrices preliminares A_tilde
        # Forma final: (batch, seq_len, p, k, k) -> (1, 500, 2, 4, 4)
        A_tilde = A_elements.view(batch_size, seq_len, self.p, self.k, self.k)
        
        # Reconstruir la matriz triangular inferior L_t
        L_t = torch.zeros(batch_size, seq_len, self.k, self.k, device=z.device)
        tril_indices = torch.tril_indices(row=self.k, col=self.k, offset=0,device=z.device)
        L_t[:, :, tril_indices[0], tril_indices[1]] = L_elements
        
        diag_idx = torch.arange(self.k)
        # Aplicamos softplus para hacer los valores positivos y sumamos 1e-4 por seguridad numérica
        L_t[:, :, diag_idx, diag_idx] = torch.nn.functional.softplus(L_t[:, :, diag_idx, diag_idx]) + 1e-4
        # Calculamos Sigma_t = L_t * L_t'
        # bmm es batch matrix multiplication
        L_t_transposed = L_t.transpose(-1, -2)
        Sigma_t = torch.matmul(L_t, L_t_transposed)
        A1_causal, A2_causal = self.enforce_causality(A_tilde, L_t)
        
        return A1_causal, A2_causal, Sigma_t, L_t
        

    def enforce_causality(self, A_tilde, L_t):
        """
        Aplica la transformación de Ansley-Kohn para p=2.
        A_tilde: Tensor de forma (batch, seq_len, p, k, k)
        L_t: Factor de Cholesky de la matriz de covarianza, forma (batch, seq_len, k, k)
        """
        batch_size, seq_len, _, k, _ = A_tilde.size()
        device = A_tilde.device
        
        # Matriz Identidad repetida para batch y seq_len
        
        I = torch.eye(k, device=device).view(1, 1, k, k).expand(batch_size, seq_len, k, k)
        jitter = 1e-5 * I
        #Matrices de autocorrelación parcial (P_1, P_2)
        A1_tilde = A_tilde[:, :, 0, :, :]
        A2_tilde = A_tilde[:, :, 1, :, :]
        
        # Para P1
        # I + A1_tilde * A1_tilde'
        mat1 = I + torch.matmul(A1_tilde, A1_tilde.transpose(-1, -2)) + jitter
        B1 = torch.linalg.cholesky(mat1)
        P1 = torch.matmul(torch.linalg.inv(B1), A1_tilde)
        
        # Para P2
        mat2 = I + torch.matmul(A2_tilde, A2_tilde.transpose(-1, -2))+ jitter
        B2 = torch.linalg.cholesky(mat2)
        P2 = torch.matmul(torch.linalg.inv(B2), A2_tilde)
        
        # Recursión Ansley-Kohn para p=2
        # Inicialización (s=0)
        # Como L_0 y L_0* son identidades, A_{1,1} es simplemente P1
        A_1_1 = P1
        A_1_1_star = P1.transpose(-1, -2)
        
        Sigma_1 = I - torch.matmul(A_1_1, A_1_1.transpose(-1, -2)) + jitter
        Sigma_1_star = I - torch.matmul(A_1_1_star, A_1_1_star.transpose(-1, -2)) + jitter
        
        L_1 = torch.linalg.cholesky(Sigma_1)
        L_1_star = torch.linalg.cholesky(Sigma_1_star)
        
        # Iteración (s=1)
        # A_{2,2} = L_1 * P_2 * (L_1*)^{-1}
        inv_L_1_star = torch.linalg.inv(L_1_star)
        A_2_2 = torch.matmul(torch.matmul(L_1, P2), inv_L_1_star)
        
        # A*_{2,2} = L_1* * P_2' * (L_1)^{-1}
        inv_L_1 = torch.linalg.inv(L_1)
        A_2_2_star = torch.matmul(torch.matmul(L_1_star, P2.transpose(-1, -2)), inv_L_1)
        
        # A_{2,1} = A_{1,1} - A_{2,2} * A*_{1,1}
        A_2_1 = A_1_1 - torch.matmul(A_2_2, A_1_1_star)
        
        # Sigma_2 = Sigma_1 - A_{2,2} * Sigma_1* * A_{2,2}'
        term_sigma2 = torch.matmul(torch.matmul(A_2_2, Sigma_1_star), A_2_2.transpose(-1, -2))
        Sigma_2 = Sigma_1 - term_sigma2 + jitter
        
        L_p = torch.linalg.cholesky(Sigma_2) # L_p donde p=2
        
        # Transformación Final
        # M = L_t * L_p^{-1}
        inv_L_p = torch.linalg.inv(L_p)
        M = torch.matmul(L_t, inv_L_p)
        inv_M = torch.linalg.inv(M)
        
        # A_1 = M * A_{2,1} * M^{-1}
        A1_causal = torch.matmul(torch.matmul(M, A_2_1), inv_M)
        
        # A_2 = M * A_{2,2} * M^{-1}
        A2_causal = torch.matmul(torch.matmul(M, A_2_2), inv_M)
        
        return A1_causal, A2_causal
