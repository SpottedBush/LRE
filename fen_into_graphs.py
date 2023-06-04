wthreats = []
bthreats = []

def add_threat(moves, team):
    for node in moves:
        if team == 0:
            wthreats.append(node)
        else:
            bthreats.append(node)

class Node: #node example : ["e4","R", 0, ['e1', 'e2', 'e3']]
    name = ""
    piece = "-"
    team = 0
    moves = []

def get_act_column(letter):
    if letter == 'a':
        return 1
    elif letter == 'b':
        return 2
    elif letter == 'c':
        return 3
    elif letter == 'd':
        return 4
    elif letter == 'e':
        return 5
    elif letter == 'f':
        return 6
    elif letter == 'g':
        return 7
    elif letter == 'h':
        return 8

def is_threaten(board, square, ally_team):
    if ally_team == 0: #white
        for node in bthreats:
            if node.name == square:
                return True
    if ally_team == 1: #black
        for node in wthreats:
            if node.name == square:
                return True
    return False

def add_moves_k(board, column, line):
    moves = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if not (line + i >= 0 and line + i < 8 and column + j < 8 and column + j >=0):
                continue
            if (i == 0 and j == 0) or is_threaten(board, board[line + i][column + j].name, board[line][column].team):
                continue
            moves.append(board[line + i][column + j])
    board[line][column].moves = moves
    return board

def add_moves_p(board, column, line):
    moves = []
    square = board[line][column].name
    if board[line][column].team == 1:
        if line + 1 < 8 and square[1] <= "7" and board[line + 1][column].piece == "-":
            moves.append(board[line + 1][column])
        if line + 2 < 8 and square[1] == "7" and board[line + 2][column].piece == "-":
            moves.append(board[line + 2][column])
        if line + 1 < 8 and column - 1 >= 0 and board[line + 1][column - 1].piece != "-" and board[line + 1][column - 1].team != board[line][column].team:
            moves.append(board[line + 1][column - 1])
            bthreats.append(board[line + 1][column - 1])
        if line + 1 < 8 and column + 1 < 8 and board[line + 1][column + 1].piece != "-" and board[line + 1][column + 1].team != board[line][column].team:
            moves.append(board[line + 1][column + 1])
            bthreats.append(board[line + 1][column + 1])
    else:
        if line - 1 < 8 and square[1] >= "2" and board[line - 1][column].piece == "-":
            moves.append(board[line - 1][column])
        if line - 2 < 8 and square[1] == "2":
            moves.append(board[line - 2][column])
        if line - 1 < 8 and column - 1 >= 0 and board[line - 1][column - 1].piece != "-" and board[line - 1][column - 1].team != board[line][column].team:
            moves.append(board[line - 1][column - 1])
            wthreats.append(board[line - 1][column - 1])
        if line - 1 < 8 and column + 1 < 8 and board[line - 1][column + 1].piece != "-" and board[line - 1][column + 1].team != board[line][column].team:
            moves.append(board[line - 1][column + 1])
            wthreats.append(board[line - 1][column + 1])
    board[line][column].moves = moves
    return board

def add_moves_r(board, column, line):
    moves = []
    square = board[line][column].name
    act_column = get_act_column(square[0])
    for i in range(int(square[1]) - 1, 0, -1): #below the rook
        if not (i < 8 and i >= 0):
            continue
        if board[8 - i][column].piece != "-":
            moves.append(board[8 - i][column])
            break
        moves.append(board[8 - i][column])
    for i in range(int(square[1]) + 1, 10): #above the rook
        if not (i <= 8 and i >= 0):
            continue
        if board[8-i][column].piece != "-":
            moves.append(board[8-i][column])
            break
        moves.append(board[8-i][column])
    for i in range(act_column - 2, -1, -1): #left to the rook
        if not (i < 8 and i >= 0):
            continue
        if board[line][i].piece != "-":
            moves.append(board[line][i])
            break
        moves.append(board[line][i])
    for i in range(act_column, 10): #right to the rook
        if not (i < 8 and i >= 0):
            continue
        if board[line][i].piece != "-":
            moves.append(board[line][i])
            break
        moves.append(board[line][i])
    add_threat(moves, board[line][column].team)
    board[line][column].moves = moves
    return board

def add_moves_n(board, column, line):
    moves = []
    if line + 2 < 8 and column + 1 < 8:
        moves.append(board[line + 2][column + 1])
    if line - 2 >= 0 and column + 1 < 8:
        moves.append(board[line - 2][column + 1])
    if line + 2 < 8 and column - 1 >= 0:
        moves.append(board[line + 2][column - 1])
    if line - 2 >= 0 and column - 1 >= 0:
        moves.append(board[line - 2][column - 1])
    if line + 1 < 8 and column + 2 < 8:
        moves.append(board[line + 1][column + 2])
    if line - 1 >= 0 and column + 2 < 8:
        moves.append(board[line - 1][column + 2])
    if line + 1 < 8 and column - 2 >= 0:
        moves.append(board[line + 1][column - 2])
    if line - 1 >= 0 and column - 2 >= 0:
        moves.append(board[line - 1][column - 2])
    add_threat(moves, board[line][column].team)
    board[line][column].moves = moves
    return board

def add_moves_b(board, column, line):
    moves = []
    for i in range(1, 10): #upper right
        if line + i >= 8 or column + i >= 8:
            break
        if board[line + i][column + i].piece != '-':
            moves.append(board[line + i][column + i])
            break
        moves.append(board[line + i][column + i])
    for i in range(1, 10): #lower right
        if line - i < 0 or column + i >= 8:
            break
        if board[line - i][column + i].piece != '-':
            moves.append(board[line - i][column + i])
            break
        moves.append(board[line - i][column + i])
    for i in range(1, 10): #lower left
        if line + i >= 8 or column - i < 0:
            break
        if board[line + i][column - i].piece != '-':
            moves.append(board[line + i][column - i])
            break
        moves.append(board[line + i][column - i])
    for i in range(1, 10): #lower left
        if line - i < 0 or column - i < 0:
            break
        if board[line - i][column - i].piece != '-':
            moves.append(board[line - i][column - i])
            break
        moves.append(board[line - i][column - i])
    add_threat(moves, board[line][column].team)
    board[line][column].moves = moves
    return board

def add_moves_q(board, column, line):
    board = add_moves_b(board, column, line)
    moves = board[line][column].moves
    board = add_moves_r(board, column, line)
    for node in moves:
        board[line][column].moves.append(node)
    return board

def printer(board):
    print("Board is :")
    print("  ---------------------------------")
    for j in range(len(board)):
        print( 8 - j,"| ", end = "")
        for i in range(len(board[j])):
            print(board[j][i].piece, end = " | ")
        print("\n  ---------------------------------")
    print("    a   b   c   d   e   f   g   h")

def print_moves(board):
    for line in board:
        for node in line:
            if len(node.moves) != 0:
                print(node.name, ": ", end = "")
            for move in node.moves:
                print(move.name, " ", end = "")
            if len(node.moves) != 0:
                print("")

def print_moves_piece(board, piece, piece2):
    for line in board:
        for node in line:
            if node.piece == piece or node.piece == piece2:
                if len(node.moves) != 0:
                    print(node.name, ": ", end = "")
                for move in node.moves:
                    print(move.name, " ", end = "")
                if len(node.moves) != 0:
                    print("")

def print_nodes(board):
    for line in board:
        for node in line:
            print(node.name, end =",")

def fill_up_board():
    board = [[Node() for j in range(8)] for i in range(8)]
    #Node board[7][8] is used for whose turn to play
    for i in range(len(board)):
        for j in range(len(board[i])):
            if j == 0:
                board[i][j].name = "a" + str(8 - i)
            if j == 1:
                board[i][j].name = "b" + str(8 - i)
            if j == 2:
                board[i][j].name = "c" + str(8 - i) 
            if j == 3:
                board[i][j].name = "d" + str(8 - i)
            if j == 4:
                board[i][j].name = "e" + str(8 - i)
            if j == 5:
                board[i][j].name = "f" + str(8 - i)
            if j == 6:
                board[i][j].name = "g" + str(8 - i) 
            if j == 7:
                board[i][j].name = "h" + str(8 - i)
    return board
            
#rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
#node example : ["e4","R", 0, ['e1', 'e2', 'e3']]
def fen_into_graph(fen):
    has_ended = False
    new_node = "" #whose turn to play
    column = 0
    line = 0
    board = fill_up_board()
    for letter in fen:
        if line >= 8:
            break;
        if column >= 8:
            column = 0
        if letter == ' ':
            has_ended = True
            continue
        if not has_ended:
            if letter.isdigit():
                column += int(letter)
                continue
            if letter == '/':
                line += 1
                continue
            board[line][column].piece = letter.upper()
            if letter >= 'A' and letter <= 'Z':
                board[line][column].team = 0
            if letter >= 'a' and letter <= 'z':
                board[line][column].team = 1
            column += 1
        else:
            new_node = letter # TODO: ADD THE END OF THE FEN 
    kwline = 0
    kwcol = 0
    kbline = 0
    kbcol = 0
    for line2 in range(len(board)):
        for column2 in range(len(board[0])): 
            letter = board[line2][column2].piece
            if letter == '-':
                continue
            if letter >= 'A' and letter <= 'Z':
                if letter == 'R':
                    board = add_moves_r(board, column2, line2)
                elif letter == 'N':
                    board = add_moves_n(board, column2, line2)
                elif letter == 'B':
                    board = add_moves_b(board, column2, line2)
                elif letter == 'Q':
                    board = add_moves_q(board, column2, line2)
                elif letter == 'K':
                    kwline = line2
                    kwcol = column2
                elif letter == 'P':
                    board = add_moves_p(board, column2, line2)
            if letter >= 'a' and letter <= 'z':
                if letter == 'r':
                    board = add_moves_r(board, column2, line2)
                elif letter == 'n':
                    board = add_moves_n(board, column2, line2)
                elif letter == 'b':
                    board = add_moves_b(board, column2, line2)
                elif letter == 'q':
                    board = add_moves_q(board, column2, line2)
                elif letter == 'k':
                    kbline = line2
                    kbcol = column2
                elif letter == 'p':
                    board = add_moves_p(board, column2, line2)
                #add kings moves at the very end
    board = add_moves_k(board, kwcol, kwline)
    board = add_moves_k(board, kbcol, kbline)
    for wmove in board[kwline][kwcol].moves: #Melee kings can't move near each others
        for bmove in board[kbline][kbcol].moves:
            if wmove.name == bmove.name:
                board[kwline][kwcol].moves.remove(wmove)
                board[kbline][kbcol].moves.remove(wmove)
    x_arr = [] # [features : [team, pawn, knight, bishop, rook, queen, king]]
    for line in board:
        for node in line:
            elm = []
            if node.piece == "p" or node.piece == "P":
                elm = [node.team, 1, 0, 0, 0, 0, 0]
            if node.piece == "N" or node.piece == "N":
                elm = [node.team, 0, 1, 0, 0, 0, 0]
            if node.piece == "b" or node.piece == "B":
                elm = [node.team, 0, 0, 1, 0, 0, 0]
            if node.piece == "r" or node.piece == "R":
                elm = [node.team, 0, 0, 0, 1, 0, 0]
            if node.piece == "q" or node.piece == "Q":
                elm = [node.team, 0, 0, 0, 0, 1, 0]
            if node.piece == "k" or node.piece == "K":
                elm = [node.team, 0, 0, 0, 0, 0, 1]
            if node.piece == "-":
                elm = [node.team, 0, 0, 0, 0, 0, 0]
            x_arr.append(elm)
    x_arr.append([-1,-1,-1,-1,-1,-1,-1]) #adding the master node
    return (x_arr, graph_creator(board))

def get_node_id(node):
    str = node.name
    if str[0] == "a":
        return int(str[1]) - 1
    elif str[0] == "b":
        return 10 + int(str[1]) - 3
    elif str[0] == "c":
        return 20 + int(str[1]) - 5
    elif str[0] == "d":
        return 30 + int(str[1]) - 7
    elif str[0] == "e":
        return 40 + int(str[1]) - 9
    elif str[0] == "f":
        return 50 + int(str[1]) - 11
    elif str[0] == "g":
        return 60 + int(str[1]) - 13
    elif str[0] == "h":
        return 70 + int(str[1]) - 15

def graph_creator(board): #Quick reminder : board is a matrix of nodes
    tensor = []
    master_node = 64
    for line in board:
        for node in line:
            node_id = get_node_id(node)
            tensor.append([master_node, node_id])
            for move in node.moves:
                tensor.append([node_id, get_node_id(move)])
    return tensor