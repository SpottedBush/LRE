#PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

def writing_in(name):
    f = open("Dataset_puzzles/" + name, 'a');
    f.writelines(fline)
    f.close()

for fline in open('lichess_db_puzzle.csv'):
    idx = 0
    nb_coma = 0
    while nb_coma != 7:
        if fline[idx] == ',':
            nb_coma += 1
        idx += 1
    #idx is the beginning of the theme part
    #end will be the end of the theme part
    end = 1
    while fline[idx + end] != ',':
        if fline[idx + end] == ' ':
            writing_in(fline[idx:idx + end])
            idx = idx + end + 1
            end = 1
            continue
        end += 1
    idx = 0