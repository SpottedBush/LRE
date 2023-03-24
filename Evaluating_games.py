import chess
from stockfish import Stockfish
stockfish = Stockfish(path="stockfish_15.1_win_x64_avx2\stockfish-windows-2022-x86-64-avx2.exe", depth=18)
board = chess.Board()
def blunders_in_a_game(movetext, color): #-> list[fen_str]
#movetext is a list of move, color == 0 -> white, == 1 -> black
    prev_eval = stockfish.get_evaluation().get("value")
    actual_eval = 0
    res = []
    for i in range(len(movetext)):
        if (i == 1):
            print(board.fen())
        if color == 0: #color is white
            prev_eval = stockfish.get_evaluation().get("value")
            board.push_san(str(movetext[i].white))
            stockfish.set_fen_position(board.fen())
            actual_eval = stockfish.get_evaluation().get("value")
            if actual_eval - prev_eval < -250:
                res.append(stockfish.get_fen_position())
            board.push_san(str(movetext[i].black))
            stockfish.set_fen_position(board.fen())
        else: #color is black
            board.push_san(str(movetext[i].white))
            stockfish.set_fen_position(board.fen())
            prev_eval = stockfish.get_evaluation().get("value")
            board.push_san(str(movetext[i].black))
            stockfish.set_fen_position(board.fen())
            actual_eval = stockfish.get_evaluation().get("value")
            if actual_eval - prev_eval > 250:
                res.append(stockfish.get_fen_position())
    return res
    

#get_graph_list(path/to/pgn, name) -> (snapshot[(m - 1, m)])
#   for game in pgn:
#       for blunder in blunders_in_a_game(game.moves, color):
#           make graph of snapshot - 1 and snapshot (for color)
#           add (graph - 1, graph) to res
#   return res
#example :
#edges : (6, [(k,r),(n,Q), (b,p)])
#make_graph_with_snapshot()
#

from pgn_parser import parser, pgn
f = open("lichess_pgn_2023.03.16_SpottedBush_vs_Gambler1987.1gbPYimh.pgn")
buffer = ""
for i in range(19):
    buffer = f.readline()
game = parser.parse(buffer, actions=pgn.Actions())
res = blunders_in_a_game(game.movetext , 0)
for i in res:
    print("blunder: ", i)