# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:09:15 2022

@author: vincent
"""
import lichess.api
user = lichess.api.user('thibault')
print(user['perfs']['blitz']['rating'])
from lichess.format import PYCHESS
game = lichess.api.game('Qa7FJNk2', format=PYCHESS)
print(game.end().board())
#from lichess.format import SINGLE_PGN
#pgn = lichess.api.user_games('thibault', max=200, format=SINGLE_PGN)
#with open('last200.pgn', 'w') as f:
#    f.write(pgn)
#print("ff")
#import chess
#import chess.pgn
#pgn = open("last200.pgn")
#game = chess.pgn.read_game(pgn)
#print(game.board())

from pgn_parser import parser, pgn
f = open("last200.pgn")
buffer = ""
for i in range(18):
    buffer = f.readline()
game = parser.parse(buffer, actions=pgn.Actions())
print(game.movetext[0].black)
print(game.movetext[1])
