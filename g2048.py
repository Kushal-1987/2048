import numpy as np
moveitmoveit={'right':0, 'up':1, 'left':2, 'down':3}
prob_of_2generation=0.9

def make_board():
    i,j=np.random.choice(np.arange(16),size=2,replace=False)
    x=np.zeros(16)
    x[i]=2
    x[j]=2
    score=[0, ]
    return x.reshape(4,4).copy(),score

def valid_step(board):
    valid_moves = []
    for mv in range(4):
        temp = board.copy()
        temp_score=[0, ]
        if mv==0:
            move_right(temp,temp_score)
        elif mv==1:
            move_up(temp,temp_score)
        elif mv==2:
            move_left(temp,temp_score)
        elif mv==3:
            move_down(temp,temp_score)
        if not np.array_equal(temp,board):
            valid_moves.append(mv)
    return valid_moves



def add_tile(board, prob=prob_of_2generation):
    x=np.random.choice([2,4],p=[prob, 1-prob])
    f=np.where(board.ravel() == 0)[0]
    # if f.size==0:
    #     raise KeyError
    y=np.random.choice(f)
    board.ravel()[y]=x
    
    
def move_left(board, score):
    for row in board:

        new_row = [num for num in row if num != 0]
        i = 0
        while i < len(new_row) - 1:
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                score[0] += new_row[i]
                new_row[i + 1] = 0
                i += 2
            else:
                i += 1

        new_row = [num for num in new_row if num != 0]
        new_row += [0] * (4 - len(new_row))
        row[:] = new_row

        
def move_right(board, score):
    for row in board:
        new_row = [num for num in row if num != 0]
        i = len(new_row) - 1
        while i > 0:
            if new_row[i] == new_row[i - 1]:
                new_row[i] *= 2
                score[0] += new_row[i]
                new_row[i - 1] = 0
                i -= 2
            else:
                i -= 1

        new_row = [num for num in new_row if num != 0]
        new_row = [0] * (4 - len(new_row)) + new_row
        row[:] = new_row

        

def move_up(board,score):
    move_left(board.T,score)

def move_down(board,score):
    move_right(board.T,score)
    
def move(board,mv,score):
    if mv==0:
        move_right(board,score)
    elif mv==1:
        move_up(board,score)
    elif mv==2:
        move_left(board,score)
    elif mv==3:
        move_down(board,score)
    add_tile(board)





# board,score=make_board()


# for i in range(1000):
#     steps=valid_step(board)
#     if steps==[]:
#         print('you lost')
#         break
#     else:
#         print(steps)
#         mv=np.random.choice(steps)
#         move(board,mv,score)
#         print(board,score)

