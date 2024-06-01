import pygame
import sys
import utility
import qLearning
import numpy as np
from minimax_alfabeta import get_melhor_movimento

# Constantes
WIDTH = 600
HEIGHT = 600
QUADRADO_TABULEIRO = WIDTH // 8
PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)
VERDE = (0, 128, 0)

# Inicialização do Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Reversi-Othello Game")

# Funções auxiliares
def desenha_tabuleiro(tabuleiro, player):
  screen.fill(VERDE)
  for linha in range(8):
      for coluna in range(8):
          pygame.draw.rect(screen, PRETO, (coluna * QUADRADO_TABULEIRO, linha * QUADRADO_TABULEIRO, QUADRADO_TABULEIRO, QUADRADO_TABULEIRO), 1)
          if utility.movimento_eh_valido(tabuleiro, linha, coluna, player) and player != 'B':
              pygame.draw.circle(screen, (128, 128, 128), (coluna * QUADRADO_TABULEIRO + QUADRADO_TABULEIRO // 2, linha * QUADRADO_TABULEIRO + QUADRADO_TABULEIRO // 2), QUADRADO_TABULEIRO // 6)

def desenha_pecas(tabuleiro):
  for linha in range(8):
      for coluna in range(8):
          if tabuleiro[linha][coluna] == 'P':
              pygame.draw.circle(screen, PRETO, (coluna * QUADRADO_TABULEIRO + QUADRADO_TABULEIRO // 2, linha * QUADRADO_TABULEIRO + QUADRADO_TABULEIRO // 2), QUADRADO_TABULEIRO // 3)
          elif tabuleiro[linha][coluna] == 'B':
              pygame.draw.circle(screen, BRANCO, (coluna * QUADRADO_TABULEIRO + QUADRADO_TABULEIRO // 2, linha * QUADRADO_TABULEIRO + QUADRADO_TABULEIRO // 2), QUADRADO_TABULEIRO // 3)


def main():
    # Instanciando a IA de Q-Learning
    agent_ai = qLearning.QLearningAgent('QTable.npy')
    player = 'P'
    ai_player = 'B'

    tabuleiro = utility.inicializar_tabuleiro()

    while not utility.game_over(tabuleiro):
        movimentos_validos = utility.get_movimentos_validos(tabuleiro, player)
        if not movimentos_validos:
            print(f"{player} não tem movimentos possíveis. Passando a vez.")
            player = 'B' if player == 'P' else 'P'
            continue
        
        if player == ai_player and movimentos_validos:
            qTableMove = agent_ai.play(tabuleiro)
            if qTableMove is None:
                best_move = get_melhor_movimento(tabuleiro, ai_player)
                if best_move is None:
                    print("Não há movimentos possíveis para a IA.")
                    player = 'B' if player == 'P' else 'P'
                    continue
                else:
                    tabuleiro = utility.fazer_jogada(tabuleiro, best_move[0], best_move[1], ai_player)
            else:
                tabuleiro = utility.fazer_jogada(tabuleiro, qTableMove[0], qTableMove[1], ai_player)
            player = 'B' if player == 'P' else 'P'
        else:
            for event in pygame.event.get():
              if event.type == pygame.QUIT:
                  pygame.quit()
                  sys.exit()
              if event.type == pygame.MOUSEBUTTONDOWN:
                  x, y = event.pos
                  linha, coluna = y // QUADRADO_TABULEIRO, x // QUADRADO_TABULEIRO
                  if utility.movimento_eh_valido(tabuleiro, linha, coluna, player):
                      tabuleiro = utility.fazer_jogada(tabuleiro, linha, coluna, player)
                      player = 'B' if player == 'P' else 'P'
        desenha_tabuleiro(tabuleiro, player)
        desenha_pecas(tabuleiro)
        pygame.display.flip()

    vencedor = utility.get_vencedor(tabuleiro)
    print("Vencedor: ", vencedor)


if __name__ == "__main__":
    main()