import numpy as np

# Definindo constante de exploration vs exploitation gamma como 0.8
gamma = 0.8

# Definindo matriz de rewards
R = [[-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]]

# Definindo matriz Q inicialmente com 0's
Q = np.array(np.zeros((6, 6)))

# 10000 iteracoes
for _ in range(10000):
    # Escolhendo um estado inicial aleatoriamente
    est_inicial = np.random.randint(0, 6)
    while True:
        # Escolhendo os indices de acoes possiveis da linha (acoes != -1)
        acoes_possiveis = [i for i in range(len(R[est_inicial])) if R[est_inicial][i] > -1]
        # Escolhendo um dos indices
        random_acao = np.random.choice(acoes_possiveis)
        # Pegando o valor Q maximo
        max_q_valor = max([i for i in Q[random_acao]])
        # Atualizando a matriz q -> Q(state, action) = R(state, action) + gamma * max(Q(state, all_actions))
        Q[est_inicial][random_acao] = R[est_inicial][random_acao] + gamma * max_q_valor
        # Novo estado se torna a acao escolhida
        est_inicial = random_acao
        # Caso essa nova acao seja o estado final (nesse caso o 5) saimos do loop
        if est_inicial == 5:
            break

# Imprimindo a matriz Q
print(Q)
