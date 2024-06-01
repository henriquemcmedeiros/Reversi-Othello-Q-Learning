import numpy as np
import utility

class QLearningAgent:
    def __init__(self, q_table_arquivo=None, episodios=10000, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q_table_arquivo = q_table_arquivo
        self.episodios = episodios
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = self.load_q_table(q_table_arquivo) if q_table_arquivo else {}

    def load_q_table(self, q_table_arquivo):
        return np.load(q_table_arquivo, allow_pickle=True).item()

    def save_q_table(self):
        np.save('QTable.npy', self.Q)

    def get_hash_tabuleiro(self, tabuleiro):
        hash_valor = 0
        for linha in tabuleiro:
            for elem in linha:
                valor = 0 if elem == ' ' else 1 if elem == 'B' else 2
                hash_valor = hash_valor * 3 + valor
        return hash_valor

    def get_maior_q(self, chave_tabuleiro):
        return max(self.Q[chave_tabuleiro].values())

    def get_acao_maximo_q(self, chave_tabuleiro):
        maior_valor = self.get_maior_q(chave_tabuleiro)
        for chave_acao, valor in self.Q[chave_tabuleiro].items():
            if valor == maior_valor:
                return chave_acao

    def play(self, tabuleiro):
        chave_tabuleiro = self.get_hash_tabuleiro(tabuleiro)

        if chave_tabuleiro not in self.Q:
            print("Trocando para Minimax, jogada não mapeada")
            return None
        else:
            action = self.get_acao_maximo_q(chave_tabuleiro)
        
        return action

    def train(self):
        for _ in range(self.episodios):
            tabuleiro = utility.inicializar_tabuleiro()
            player = 'P'

            while True:
                chave_tabuleiro = self.get_hash_tabuleiro(tabuleiro)

                if chave_tabuleiro not in self.Q:
                    self.Q[chave_tabuleiro] = {}

                acoes_validas = utility.get_movimentos_validos(tabuleiro, player)

                if not acoes_validas:
                    player = 'B' if player == 'P' else 'P'
                    acoes_validas = utility.get_movimentos_validos(tabuleiro, player)

                if np.random.uniform(0, 1) < self.epsilon:
                    action = acoes_validas[np.random.choice(len(acoes_validas))]
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