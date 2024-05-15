import numpy as np
import utility
import hashlib

class QLearningAgent:
    def __init__(self, q_table_file=None, episodes=10000, alpha=0.5, gamma=0.9, epsilon=0.1):
        if q_table_file is not None:
            self.Q = self.load_q_table(q_table_file)
        else:
            self.Q = {}
        self.q_table_file = q_table_file
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def load_q_table(self, q_table_file):
        return np.load(q_table_file, allow_pickle='TRUE').item()

    def get_hash_tabuleiro(self, tabuleiro):
        # Calcula o hash do estado do tabuleiro
        #return hashlib.sha256(str(tabuleiro).encode()).hexdigest()
        hash_valor = 0
        for i in range(8):
            for j in range(8):
                # ' ' = 0 'B' = 1 e 'P' = 2
                valor = 0 if tabuleiro[i][j] == ' ' else 1 if tabuleiro[i][j] == 'B' else 2
                hash_valor = hash_valor * 3 + valor
        return hash_valor
    
    def get_maior_q(self, chave_tabuleiro):
        valores = list(self.Q[chave_tabuleiro].values())
        return max(valores)
    
    def get_acao_maximo_q(self, chave_tabuleiro):
        maior_valor = self.get_maior_q(chave_tabuleiro)
        for chave_acao in self.Q[chave_tabuleiro].keys():
            if self.Q[chave_tabuleiro][chave_acao] == maior_valor:
                return chave_acao
    
    def play(self, tabuleiro, player):
        chave_tabuleiro = self.get_hash_tabuleiro(tabuleiro)

        # Check if the state exists in the Q-table
        if chave_tabuleiro not in self.Q:
            print("Jogada não mapeada")
            valid_actions = utility.get_movimentos_validos(tabuleiro, player)
            action = valid_actions[np.random.choice(len(valid_actions))]
        else:
        #     matriz = [self.Q[chave_tabuleiro][i:i+8] for i in range(0, len(self.Q[chave_tabuleiro]), 8)]
        #     # Imprimindo a matriz
        #     for linha in matriz:
        #         print(*linha)
        #     print("=================")
            # Pega a ação com o maior valor na tabela Q
            action = self.get_acao_maximo_q(chave_tabuleiro)

        return action

    def train(self):
        if self.q_table_file is not None:
            self.Q = self.load_q_table(self.q_table_file)
        else:
            self.Q = {}

        for episode in range(self.episodes):
            # Inicialize o estado
            tabuleiro = utility.inicializar_tabuleiro()
            player = 'P'

            while True:
                # Transforma o estado hash usar como chave no dicionário Q
                chave_tabuleiro = self.get_hash_tabuleiro(tabuleiro)

                # Inicializa o estado na tabela Q e no objeto de hashes, se necessário
                if chave_tabuleiro not in self.Q:
                    self.Q[chave_tabuleiro] = {}

                # Pega as ações possíveis
                valid_actions = utility.get_movimentos_validos(tabuleiro, player)

                if not valid_actions:
                    player = 'B' if player == 'P' else 'P'
                    valid_actions = utility.get_movimentos_validos(tabuleiro, player)

                # Escolha uma ação
                if np.random.uniform(0, 1) < self.epsilon:
                    action = valid_actions[np.random.choice(len(valid_actions))]  # Ação aleatória
                else:
                    # Ação que maximiza o valor Q se tiver mapeado
                    if self.Q[chave_tabuleiro] != {}:
                        action = self.get_acao_maximo_q(chave_tabuleiro)
                    else:
                        action = utility.get_melhor_movimento(tabuleiro, player)

                # Execute a ação
                prox_tabuleiro = utility.fazer_jogada(tabuleiro, action[0], action[1], player)
                recompensa = utility.calc_recompensa(tabuleiro, player)
                player = 'B' if player == 'P' else 'P'

                # Transforma o próximo estado em uma tupla para usar como chave no dicionário Q
                next_chave_tabuleiro = self.get_hash_tabuleiro(tabuleiro)

                if next_chave_tabuleiro not in self.Q:
                    self.Q[next_chave_tabuleiro] = {}

                # Cria o elemento na tabela caso não exista
                self.Q[chave_tabuleiro][action] = self.Q[chave_tabuleiro][action] if self.Q[chave_tabuleiro].get(action) else 0
                
                maior_valor = self.get_maior_q(chave_tabuleiro)
                
                self.Q[chave_tabuleiro][action] = (1 - self.alpha) * self.Q[chave_tabuleiro][action] + self.alpha * (recompensa + self.gamma * maior_valor)
                
                tabuleiro = prox_tabuleiro

                # Se o episódio terminou, saia do loop -- GAMEOVER
                if utility.game_over(tabuleiro):
                    #print(f"Fim do teste {episode}")
                    break

        # Salvar a tabela Q
        np.save('QTable.npy', self.Q)

# Inicializar e treinar o agente
agent = QLearningAgent('QTable.npy', episodes=1000, epsilon=1)
for i in range(1000):
    agent.train()
    print(f"===== Fim do treinamento {i} =====")