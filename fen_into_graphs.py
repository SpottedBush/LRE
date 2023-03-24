import chess

#TODO : N

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

def is_threaten(board, square, team):
    for l in range(len(board)):
        for c in range (len(board[0])):
            for node in board[l][c].moves:
                if node.name == square and node.team != team:
                    return True
    return False

def add_moves_k(board, column, line):
    moves = []
    square = board[line][column].name
    for i in range(-1, 1):
        for j in range(-1, 1):
            if i == 0 and j == 0 or is_threaten(board, square, board[line][column].team):
                continue
            moves.append(board[line][column])
    board[line][column].moves = moves
    return board

def add_moves_p(board, column, line):
    moves = []
    square = board[line][column].name
    if board[line][column].team == 1:
        if line + 1 < 8 and square[1] <= "7" and board[line + 1][column].piece == "-":
            moves.append(board[line + 1][column])
        if line + 2 < 8 and square[1] == "7":
            moves.append(board[line + 2][column])
        if line + 1 < 8 and column - 1 >= 0 and board[line + 1][column - 1].piece != "-" and board[line + 1][column - 1].team != board[line][column].team:
            moves.append(board[line + 1][column - 1])
        if line + 1 < 8 and column + 1 < 8 and board[line + 1][column + 1].piece != "-" and board[line + 1][column + 1].team != board[line][column].team:
            moves.append(board[line + 1][column + 1])
    else:
        if line - 1 < 8 and square[1] >= "2" and board[line - 1][column].piece == "-":
            moves.append(board[line - 1][column])
        if line - 2 < 8 and square[1] == "2":
            moves.append(board[line - 2][column])
        if line - 1 < 8 and column - 1 >= 0 and board[line - 1][column - 1].piece != "-" and board[line - 1][column - 1].team != board[line][column].team:
            moves.append(board[line - 1][column - 1])
        if line - 1 < 8 and column + 1 < 8 and board[line - 1][column + 1].piece != "-" and board[line - 1][column + 1].team != board[line][column].team:
            moves.append(board[line - 1][column + 1])
    board[line][column].moves = moves
    return board

def add_moves_r(board, column, line):
    moves = []
    square = board[line][column].name
    act_column = get_act_column(square[0])
    for i in range(int(square[1]) - 1, 0, -1): #above the rook
        if not (i < 8 and i >= 0):
            continue
        if board[8 - i][column].piece != "-":
            moves.append(board[8 - i][column])
            break
        moves.append(board[8 - i][column])
    for i in range(int(square[1]) + 1, 9): #below the rook
        if not (i < 8 and i >= 0):
            continue
        if board[8 - i][column].piece != "-":
            moves.append(board[8-i][column])
            break
        moves.append(board[8 - i][column])
    for i in range(act_column - 1, 0, -1): #left to the rook
        if not (i < 8 and i >= 0):
            continue
        if board[line][i - 1].piece != "-":
            moves.append(board[line][i-1])
            break
        moves.append(board[line][i-1])
    for i in range(act_column + 1, 9): #right to the rook
        if not (i < 8 and i >= 0):
            continue
        if board[line][8 - i].piece != "-":
            moves.append(board[line][i-1])
            break
        moves.append(board[line][i-1])
    board[line][column].moves = moves
    return board

def add_moves_n(board, column, line):
    moves = []
    square = board[line][column].name
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
    board[line][column].moves = moves
    return board

def add_moves_b(board, column, line):
    moves = []
    square = board[line][column].name
    act_column = get_act_column(square[0])
    for i in range(1, min(9 - act_column, 9 - line)): #upper right
        if line + i >= 8 or column + i >= 8:
            break
        if board[line + i][column + i].piece != '-':
            moves.append(board[line + i][column + i])
            break
        moves.append(board[line + i][column + i])
    for i in range(1, min(9 - act_column, 9 - line)): #lower right
        if line - i < 0 or column + i >= 8:
            break
        if board[line - i][column + i].piece != '-':
            moves.append(board[line - i][column + i])
            break
        moves.append(board[line - i][column + i])
    for i in range(1, min(9 - act_column, 9 - line)): #upper left
        if line + i >= 8 or column - i < 0:
            break
        if board[line + i][column - i].piece != '-':
            moves.append(board[line + i][column - i])
            break
        moves.append(board[line + i][column - i])
    for i in range(1, min(9 - act_column, 9 - line)): #lower left
        if line - i < 0 or column - i < 0:
            break
        if board[line - i][column - i].piece != '-':
            moves.append(board[line - i][column - i])
            break
        moves.append(board[line - i][column - i])
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
            if node.piece == "p" or node.piece == "P" or node.piece == "r" or node.piece == "R" or node.piece == "b" or node.piece == "B" or node.piece == "q" or node.piece == "Q":
                continue
            if len(node.moves) != 0:
                print(node.name, ": ", end = "")
            for move in node.moves:
                print(move.name, " ", end = "")
            if len(node.moves) != 0:
                print("")

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
            if letter >= 'A' and letter <= 'Z':
                board[line][column].team = 0
                if letter == 'R':
                    board[line][column].piece = "R"
                    column += 1
                elif letter == 'N':
                    board[line][column].piece = "N"
                    column += 1
                elif letter == 'B':
                    board[line][column].piece = "B"
                    column += 1
                elif letter == 'Q':
                    board[line][column].piece = "Q"
                    column += 1
                elif letter == 'K':
                    board[line][column].piece = "K"
                    column += 1
                elif letter == 'P':
                    board[line][column].piece = "P"
                    column += 1
                
            if letter >= 'a' and letter <= 'z':
                
                board[line][column].team = 1
                if letter == 'r':
                    board[line][column].piece = "r"
                    column += 1
                elif letter == 'n':
                    board[line][column].piece = "n"
                    column += 1
                elif letter == 'b':
                    board[line][column].piece = "b"
                    column += 1
                elif letter == 'q':
                    board[line][column].piece = "q"
                    column += 1
                elif letter == 'k':
                    board[line][column].piece = "k"
                    column += 1
                elif letter == 'p':
                    board[line][column].piece = "p"
                    column += 1
        else:
            new_node = letter
    count = 0
    for line2 in range(len(board)):
        for column2 in range(len(board[0])): 
            letter = board[line2][column2].piece
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
                    board = add_moves_k(board, column2, line2)
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
                    board = add_moves_k(board, column2, line2)
                elif letter == 'p':
                    board = add_moves_p(board, column2, line2)
                #add kings moves at the very end
    return board
board = fen_into_graph("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
printer(board)
print_moves(board)