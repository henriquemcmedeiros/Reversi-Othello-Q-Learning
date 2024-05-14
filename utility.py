import copy
import math

def inicializar_tabuleiro():
    tabuleiro = [[' ' for _ in range(8)] for _ in range(8)]
    tabuleiro[3][3] = tabuleiro[4][4] = 'B'
    tabuleiro[3][4] = tabuleiro[4][3] = 'P'
    return tabuleiro

def movimento_eh_valido(tabuleiro, linha, coluna, player):
  if tabuleiro[linha][coluna] != ' ':
      return False
  direcoes = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
  for dir_linha, dir_coluna in direcoes:
      r, c = linha + dir_linha, coluna + dir_coluna
      achou_peca_oponente = False
      while 0 <= r < 8 and 0 <= c < 8 and tabuleiro[r][c] != ' ':
          if tabuleiro[r][c] == player:
              if achou_peca_oponente:
                  return True
              else:
                  break
          else:
              achou_peca_oponente = True
          r += dir_linha
          c += dir_coluna
  return False

def get_movimentos_validos(tabuleiro, player):
  movimentos_validos = []
  for linha in range(8):
      for coluna in range(8):
          if movimento_eh_valido(tabuleiro, linha, coluna, player):
              movimentos_validos.append((linha, coluna))
  return movimentos_validos

def calc_recompensa(tabuleiro, player):
  player_score = 0
  oponente_score = 0
  for linha in tabuleiro:
    for elemento in linha:
      if elemento == player:
        player_score += 1
      elif elemento != ' ':
        oponente_score += 1
  return player_score - oponente_score

def game_over(tabuleiro):
  return len(get_movimentos_validos(tabuleiro, 'P')) == 0 and len(get_movimentos_validos(tabuleiro, 'B')) == 0

def fazer_jogada(tabuleiro, linha, coluna, player):
  tabuleiroCopy = copy.deepcopy(tabuleiro)
  tabuleiroCopy[linha][coluna] = player
  direcoes = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
  for dir_linha, dir_coluna in direcoes:
      r, c = linha + dir_linha, coluna + dir_coluna
      while 0 <= r < 8 and 0 <= c < 8 and tabuleiroCopy[r][c] != ' ' and tabuleiroCopy[r][c] != player:
          r += dir_linha
          c += dir_coluna
      if 0 <= r < 8 and 0 <= c < 8 and tabuleiroCopy[r][c] == player:
          r -= dir_linha
          c -= dir_coluna
          while r != linha or c != coluna:
              tabuleiroCopy[r][c] = player
              r -= dir_linha
              c -= dir_coluna
  return tabuleiroCopy

def get_melhor_movimento(tabuleiro, player):
  melhor_movimento = None
  melhor_heuristica = -math.inf
  movimentos_validos = get_movimentos_validos(tabuleiro, player)
  if len(movimentos_validos) == 1:
    return movimentos_validos[0]
  for movimento in movimentos_validos:
    novo_tabuleiro = [linha[:] for linha in tabuleiro]
    fazer_jogada(novo_tabuleiro, movimento[0], movimento[1], player)
    heuristica = calc_recompensa(novo_tabuleiro, player)
    if heuristica > melhor_heuristica:
      melhor_heuristica = heuristica
      melhor_movimento = movimento
  return melhor_movimento