import matplotlib.pyplot as plt
import networkx as nx
import torch
from pgn_parser import parser, pgn
import os

from evaluating_games import blunders_in_a_game

from fen_Into_Graphs import fen_into_graph
from fen_Into_Graphs import printer
from fen_Into_Graphs import print_moves
from fen_Into_Graphs import graph_creator

class Node: #node example : ["e4","R", 0, ['e1', 'e2', 'e3']]
    name = ""
    piece = "-"
    team = 0
    moves = []

# visualization graph
def plot_graph(edge_index):
    # Create a directed graph
    graph = nx.DiGraph()

    # Add edges to the graph
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        src, dst = edge_index[:, i]
        graph.add_edge(src.item(), dst.item())

    # Draw the graph
    pos = nx.spring_layout(graph, seed=42)  # Positions of the nodes
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    plt.show()

# Main :
path = os.path.join('dataset', 'Dataset_parties')
path = os.path.join(path, "Lerelliz.pgn")
f = open(path)
user = path[24:-4]
buffer = f.readline()
dataset = []
team = 0 # White is default
while buffer != "":
    # Reaching the team line
    if user in buffer: 
        if buffer[1:6] == "Black":
            team = 1
        if buffer[1:6] == "White":
            team = 0
        # Reaching the end of a game in pgn format
    if buffer[0] == '1':
        game = parser.parse(buffer, actions=pgn.Actions())
        print("game movetext:", game.movetext)
        blunders = blunders_in_a_game(game.movetext , team)
        for blunder in blunders:
            print("blunder: ", blunder)
            tensor = fen_into_graph(blunder)
            # Each element of dataset is a tensor meaning a whole graph
            dataset.append(torch.tensor(tensor))
    buffer = f.readline()

