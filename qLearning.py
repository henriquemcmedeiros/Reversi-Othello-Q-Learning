import numpy as np
import utility
import os

class QLearningAgent:
    def __init__(self, q_table_file=None, episodes=10000, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q_table_file = q_table_file
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = self.load_q_table(q_table_file) if q_table_file else {}

    def load_q_table(self, q_table_file):
        return np.load(q_table_file, allow_pickle=True).item()

    def save_q_table(self):
        np.save('QTable.npy', self.Q)

    def get_hash_tabuleiro(self, tabuleiro):
        hash_valor = 0
        for row in tabuleiro:
            for cell in row:
                valor = 0 if cell == ' ' else 1 if cell == 'B' else 2
                hash_valor = hash_valor * 3 + valor
        return hash_valor

    def get_maior_q(self, chave_tabuleiro):
        return max(self.Q[chave_tabuleiro].values())

    def get_acao_maximo_q(self, chave_tabuleiro):
        maior_valor = self.get_maior_q(chave_tabuleiro)
        for chave_acao, valor in self.Q[chave_tabuleiro].items():
            if valor == maior_valor:
                return chave_acao

    def play(self, tabuleiro, player):
        chave_tabuleiro = self.get_hash_tabuleiro(tabuleiro)

        if chave_tabuleiro not in self.Q:
            print("Jogada não mapeada")
            valid_actions = utility.get_movimentos_validos(tabuleiro, player)
            action = valid_actions[np.random.choice(len(valid_actions))]
        else:
            action = self.get_acao_maximo_q(chave_tabuleiro)
        
        return action

    def train(self):
        for episode in range(self.episodes):
            tabuleiro = utility.inicializar_tabuleiro()
            player = 'P'

            while True:
                chave_tabuleiro = self.get_hash_tabuleiro(tabuleiro)

                if chave_tabuleiro not in self.Q:
                    self.Q[chave_tabuleiro] = {}

                valid_actions = utility.get_movimentos_validos(tabuleiro, player)

                if not valid_actions:
                    player = 'B' if player == 'P' else 'P'
                    valid_actions = utility.get_movimentos_validos(tabuleiro, player)

                if np.random.uniform(0, 1) < self.epsilon:
                    action = valid_actions[np.random.choice(len(valid_actions))]
                else:
                    action = self.get_acao_maximo_q(chave_tabuleiro) if self.Q[chave_tabuleiro] else utility.get_melhor_movimento(tabuleiro, player)

                prox_tabuleiro = utility.fazer_jogada(tabuleiro, action[0], action[1], player)
                recompensa = utility.calc_recompensa(tabuleiro, player)
                player = 'B' if player == 'P' else 'P'

                next_chave_tabuleiro = self.get_hash_tabuleiro(tabuleiro)

                if next_chave_tabuleiro not in self.Q:
                    self.Q[next_chave_tabuleiro] = {}

                if action not in self.Q[chave_tabuleiro]:
                    self.Q[chave_tabuleiro][action] = 0

                maior_valor = self.get_maior_q(next_chave_tabuleiro) if self.Q[next_chave_tabuleiro] else 0

                self.Q[chave_tabuleiro][action] = (1 - self.alpha) * self.Q[chave_tabuleiro][action] + self.alpha * (recompensa + self.gamma * maior_valor)

                tabuleiro = prox_tabuleiro

                if utility.game_over(tabuleiro):
                    break

        self.save_q_table()

# Initialize and train the agent
#agent = QLearningAgent('QTable.npy', episodes=1000, epsilon=1)
#for i in range(1000):
#    agent.train()
#    file_size = os.path.getsize('QTable.npy')
#    print("Size of the file:", file_size / 1_000_000, "MB")
#    print(f"===== Fim do treinamento {i} =====")
