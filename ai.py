from g2048 import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  
        )

    def forward(self, x):
        return self.fc(x)


def board_to_tensor(board):
    # Log scaling is often better for 2048 boards
    b = board.copy()
    b[b == 0] = 1
    return torch.tensor(np.log2(b), dtype=torch.float32).unsqueeze(0).view(1, 16)

def next_states(original_board, mv, prob=prob_of_2generation):
    temp_score=[0,]
    board=original_board.copy()
    if mv==0:
        move_right(board,temp_score)
    elif mv==1:
        move_up(board,temp_score)
    elif mv==2:
        move_left(board,temp_score)
    elif mv==3:
        move_down(board,temp_score)
    free=np.where(board.ravel() == 0)[0]
    n=len(free)
    next_boards=[]
    prob_board=[]
    for i in free:
        for val, p in ((2, prob), (4, 1 - prob)):
            b = board.copy()
            b.ravel()[i] = val
            next_boards.append(board_to_tensor(b))
            prob_board.append(p / n)
    return next_boards, prob_board

# def expected_state_value(board, model):
#     value=[]
#     valid_moves=valid_step(board)
#     for mv in range(4):
#         if mv in valid_moves:
#             nxt_states, probs = next_states(board, mv)
#             with torch.no_grad():
#                 vals = model(nxt_states).view(-1)
#             value.append(torch.dot(probs, vals).item())
#         else:
#             value.append(-1e10)
#     value=np.array(value)
#     return max(value), np.argmax(value)

model = DQN()
target_model = DQN()
target_model.load_state_dict(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

gamma = 0.99
epsilon = 1.0
min_epsilon = 0.05
decay = 0.995
memory = deque()
batch_size = 256
exploration=True

for episode in range(10):
    board, score = make_board()
    done = False
    while True:
        state = board_to_tensor(board)
        old_score=score[0]
        valid_moves =valid_step(board)
        
        #action 
        if done:
            print(score[0])
            break
        elif exploration and random.random() < epsilon:
            action = random.choice(valid_moves)
        else:
            with torch.no_grad():
                q_values = model(state).squeeze()
                for m in range(4):
                    if m not in valid_moves:
                        q_values[m] = -float('inf')  # mask invalid moves
                action = torch.argmax(q_values).item()
        
        move(board, action, score)
        
        reward= score[0]-old_score
        new_state=board_to_tensor(board)
        done=(len(valid_step(board))==0)
        
        memory.append((state, action, reward, new_state, done))
        if len(memory)>=10000:
            memory.popleft()
        if done:
            print(f"Episode {episode}, Score: {score[0]}, Max Tile: {board.max()}")

            
        
    if len(memory)<batch_size:
        continue
    batch = random.sample(memory, batch_size)
    
    states = torch.cat([s for s, _, _, _, _ in batch])
    actions = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.int64)
    rewards = torch.tensor([y for _, _, y, _, _ in batch], dtype=torch.float32)
    # new_states  = torch.tensor([s for _, _, _, s, _ in batch], dtype=torch.float32)
    dones=torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32)
    target_list=[]
    for state, action, reward, next_board, done in batch:
        valid_actions = valid_step(next_board)
        values = []
        for a in valid_actions:
            future_states, probs = next_states(next_board, a)
            with torch.no_grad():
                q_vals = target_model(torch.cat(future_states)).max(dim=1)[0]
            expected_q = torch.dot(torch.tensor(probs), q_vals)
            values.append(expected_q)
        if len(values) > 0:
            max_expected_q = max(values)
        else:
            max_expected_q = 0.0
        target = reward + gamma * (1 - done) * max_expected_q
        target_list.append(target)

    targets = torch.tensor(target_list, dtype=torch.float32)
    q_preds = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    q_preds = torch.clamp(q_preds, -20, 20)
    targets = torch.clamp(targets, -20, 20)
    loss = criterion(q_preds, targets.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())
    epsilon = max(min_epsilon, epsilon * decay)
                                                                