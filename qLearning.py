import numpy as np
import utility
import hashlib

class QLearningAgent:
    def __init__(self, q_table_file=None, episodes=10000, alpha=0.5, gamma=0.9, epsilon=0.1):
        if q_table_file is not None:
            self.Q = self.load_q_table(q_table_file)
        else:
            self.Q = {}
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_hashes = {}  # Armazena os hashes dos estados do tabuleiro

    def load_q_table(self, q_table_file):
        Q = np.load(q_table_file, allow_pickle='TRUE').item()
        return Q

    def get_state_hash(self, tabuleiro):
        # Calcula o hash do estado do tabuleiro
        return hashlib.sha256(str(tabuleiro).encode()).hexdigest()
    
    def play(self, tabuleiro, player):
        # Get the hash of the current state
        chave_tabuleiro = self.get_state_hash(tabuleiro)

        # Check if the state exists in the Q-table
        if chave_tabuleiro not in self.Q:
            valid_actions = utility.get_movimentos_validos(tabuleiro, player)
            action = valid_actions[np.random.choice(len(valid_actions))]
        else:
            # Get the action that has the maximum Q-value
            max_action_value = np.argmax(self.Q[chave_tabuleiro])
            action = (max_action_value // 8, max_action_value % 8)

        return action

    def train(self):
        for episode in range(self.episodes):
            # Inicialize o estado
            tabuleiro = utility.inicializar_tabuleiro()
            player = 'P'

            while True:
                # Transforma o estado em uma tupla para usar como chave no dicionário Q
                #chave_tabuleiro = tuple(map(tuple, tabuleiro))
                chave_tabuleiro = self.get_state_hash(tabuleiro)

                # Inicializa o estado na tabela Q e no objeto de hashes, se necessário
                if chave_tabuleiro not in self.Q:
                    self.Q[chave_tabuleiro] = np.zeros(8*8)
                    self.state_hashes[chave_tabuleiro] = self.get_state_hash(tabuleiro)

                # Obtenha as ações possíveis
                valid_actions = utility.get_movimentos_validos(tabuleiro, player)

                if not valid_actions:
                    #print(f"{player} não tem movimentos possíveis. Passando a vez.")
                    player = 'B' if player == 'P' else 'P'
                    valid_actions = utility.get_movimentos_validos(tabuleiro, player)

                # Escolha uma ação
                if np.random.uniform(0, 1) < self.epsilon:
                    action = valid_actions[np.random.choice(len(valid_actions))]  # Ação aleatória
                else:
                    # Ação que maximiza o valor Q
                    action_values = self.Q[chave_tabuleiro][valid_actions]
                    action = utility.get_melhor_movimento(tabuleiro, player)

                position_action = action[0] * 8 + action[1]

                # Execute a ação
                prox_tabuleiro = utility.fazer_jogada(tabuleiro, action[0], action[1], player)
                recompensa = utility.calc_recompensa(tabuleiro, player)
                player = 'B' if player == 'P' else 'P'

                # Transforma o próximo estado em uma tupla para usar como chave no dicionário Q
                next_chave_tabuleiro = self.get_state_hash(tabuleiro)

                if next_chave_tabuleiro not in self.Q:
                    self.Q[next_chave_tabuleiro] = np.zeros(8*8)
                    self.state_hashes[next_chave_tabuleiro] = self.get_state_hash(tabuleiro)

                self.Q[chave_tabuleiro][position_action] = (1 - self.alpha) * self.Q[chave_tabuleiro][position_action] + \
                                                  self.alpha * (recompensa + self.gamma * np.max(self.Q[next_chave_tabuleiro]))

                tabuleiro = prox_tabuleiro

                # Se o episódio terminou, saia do loop -- GAMEOVER
                if utility.game_over(tabuleiro):
                    print(f"Fim do teste {episode}")
                    break

        # Salvar a tabela Q
        np.save('QTable.npy', self.Q)

# Inicializar e treinar o agente - 'hash_set.txt'
agent = QLearningAgent('QTable.npy')
agent.train()