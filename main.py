import time
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

# Inicialização tabela Q
Q = np.zeros((8, 8, 8*8))

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

def contar_pecas(tabuleiro):
    PRETO_count = sum(linha.count('P') for linha in tabuleiro)
    BRANCO_count = sum(linha.count('B') for linha in tabuleiro)
    return PRETO_count, BRANCO_count

def get_vencedor(tabuleiro):
    PRETO_count, BRANCO_count = contar_pecas(tabuleiro)
    if PRETO_count > BRANCO_count:
      return 'PRETO'
    elif BRANCO_count > PRETO_count:
        return 'BRANCO'
    else:
        return 'EMPATE'

def main():
  tabuleiro = utility.inicializar_tabuleiro()

  # Instanciando a IA
  agent_ai = qLearning.QLearningAgent('QTable.npy')

  player = 'P'
  ai_player = 'B'

  while not utility.game_over(tabuleiro):
      movimentos_validos = utility.get_movimentos_validos(tabuleiro, player)
      if not movimentos_validos:
          print(f"{player} não tem movimentos possíveis. Passando a vez.")
          player = 'B' if player == 'P' else 'P'
          continue
      
      if player == ai_player and movimentos_validos:
          best_move = agent_ai.play(tabuleiro, ai_player)
          if best_move is None:
              print("Não há movimentos possíveis para o jogador AI.")
              player = 'B' if player == 'P' else 'P' 
              continue
          utility.fazer_jogada(tabuleiro, best_move[0], best_move[1], ai_player)
          player = 'P'
          continue
      else:
          for event in pygame.event.get():
              if event.type == pygame.QUIT:
                  pygame.quit()
                  sys.exit()
              if event.type == pygame.MOUSEBUTTONDOWN:
                  x, y = event.pos
                  linha, coluna = y // QUADRADO_TABULEIRO, x // QUADRADO_TABULEIRO
                  if utility.movimento_eh_valido(tabuleiro, linha, coluna, player):
                      utility.fazer_jogada(tabuleiro, linha, coluna, player)
                      player = 'B' if player == 'P' else 'P'
      desenha_tabuleiro(tabuleiro, player)
      desenha_pecas(tabuleiro)
      pygame.display.flip()

  vencedor = get_vencedor(tabuleiro)
  print("Vencedor: ", vencedor)

if __name__ == "__main__":
  main()