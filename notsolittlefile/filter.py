#open('filtered','w').writelines([ line for line in open('last200.pgn') if 'thibault' in line])


def ltoi(intlist): #list to int
    res = 0
    for i in range(len(intlist)):
        res += intlist[i] * 10 ** (len(intlist) - i - 1)
    return res

def has_name(strs, name):
    return (name in strs[2] or name in strs[3])

def writing_in(name, strs):
    f = open(name + '.pgn', 'a');
    f.writelines(line for line in strs)
    f.writelines(fline)
    f.close()

strs = ["" for x in range(18)];
for fline in open('Dataset_parties/lichess_db_standard_rated_2013-01.pgn'):
    actual_line = 0
    if ("Event" in fline):
        for i in range(len(strs)):
            strs[i] = ""
        strs[0] = fline
        continue
    for i in range(len(strs)):
        if strs[i] == "":
            actual_line = i
            break;
            
    if actual_line == len(strs) - 1:
        elow = ltoi([int(i) for i in strs[7] if i.isdigit()])
        elob = ltoi([int(i) for i in strs[8] if i.isdigit()])
    
        if ("Classical" in strs[0] and 1700 >= int(elow) >= 1300 and 1700 >= int(elob) >= 1300 and "600+0" in strs[13]):
            writing_in(strs[2][8:-3] ,strs)
            writing_in(strs[3][8:-3] ,strs)
        for i in range(len(strs)):
            strs[i] = ""
        continue
    
    strs[actual_line] = fline