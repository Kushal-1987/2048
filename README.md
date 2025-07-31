# Mastering 2048 with Reinforcement Learning 

This repository documents the first three weeks of an 8‑week project to build an AI agent that can play **2048**, using reinforcement learning and neural networks.

---
## Problem Statement

2048 is a popular single-player puzzle game in which players combine tiles with matching values to reach the 2048 tile. Despite its simple interface, it involves strategic decision-making and randomness, making it a good candidate for reinforcement learning.

The objective of this project is to create an intelligent agent that can consistently win the game by reaching at least the 2048 tile, or higher (such as 4096 or 8192), using AI and ML techniques.

**Goal:** Design and compare two AI agents using (1) n-Tuple networks and (2) Deep Q-Learning to play 2048 effectively.

## Proposed Solutions

### 1. n-Tuple Network

The n-Tuple network is a feature-based function approximation technique that uses predefined patterns (tuples) of board positions to evaluate a state.

- The 4x4 game board is divided into overlapping tuples such as rows, columns, or shapes.
- Each tuple has a lookup table (LUT) to store the value for specific tile combinations.
- The value of the board is the sum of values from all active tuples.
- The tables are updated using Temporal Difference (TD) learning, such as TD(0).
- This method is fast and interpretable, requiring low computational resources.

### 2. Deep Q-Learning (DQN)

Deep Q-Learning uses a neural network to approximate the Q-function, which estimates the expected future reward for each action.

- The board is encoded as a matrix input, and the actions are: up, down, left, and right.
- The network learns Q-values `Q(s, a)` using the Bellman equation.
- Experience Replay and Target Networks are used to stabilize training.
- An ε-greedy policy is used to balance exploration and exploitation.
- With enough training, the model can learn complex strategies and reach very high tiles.

## Expected Outcomes

- A trained n-Tuple agent that plays the game quickly and consistently.
- A Deep Q-Learning agent that improves over episodes and learns high-reward strategies.
- Evaluation based on average score, win rate, and maximum tile achieved.

---

## Week 1: Reinforcement Learning Foundations  
**Goal:** Establish a solid understanding of RL concepts and their mathematical foundations.

### Topics Covered:
- **Markov Decision Processes (MDP)**
- **Bellman Equation** (derivations & intuition)
- **Temporal Difference (TD) Learning** (especially TD(0))
- **Q‑Learning** (off‑policy control)
- **Artificial Neural Networks** (structure, activation, backprop)
- **Intro to Deep Q‑Learning** (how DQN extends Q‑Learning)

### Resources:

- **Markov Decision Processes (MDP):** https://youtu.be/2iF9PRriA7w  
- **Bellman Equation:** https://youtu.be/9JZID-h6ZJ0  
- **Temporal Difference (TD) Learning:** https://youtu.be/uJX_72MnKg8  
- **Q-Learning:** https://youtu.be/TiAXhVAZQl8  
- **Artificial Neural Networks (ANN):** https://youtu.be/WuuY2V475Yg  
- **Foundations of Deep Q-Learning:** https://youtu.be/_R8TufWrQyY  
- **Deep Reinforcement Learning Basics:** https://youtu.be/iKdlKYG78j4  

---

## Week 2: Deep Q‑Networks & PyTorch  
**Goal:** Learn how to build and train neural networks, and apply them in a Deep Q‑Network for RL.

### Topics Covered:
- **Gradient Descent** & **Backpropagation**
- **Neural Network Architecture** (layers, activations)
- **PyTorch Basics** (tensors, autograd, model definition)
- **Key DQN Concepts:**
  - Exploration vs Exploitation
  - Experience Replay Buffer
  - Training loop for DQN agent

### Resources:
- PyTorch tutorial video: https://youtu.be/OIenNRt2bjg
- Q neural networks: https://youtu.be/mo96Nqlo1L8?si=eiwLvoKcLBxBOKRk
- https://youtu.be/wrBUkpiRvCA
- https://youtu.be/Bcuj2fTH4_4
- https://youtu.be/0bt0SjbS3xc
- https://youtu.be/xVkPh9E9GfE
  

---

## Week 3: 2048 Game Mechanics & Strategies  
**Goal:** Fully understand the gameplay mechanics of 2048 and existing winning strategies.

### Topics Covered:
- **Game Rules:** tile merging, board size, scoring
- **Strategies:**
  - Edge stacking
  - Snake pattern
  - Maximizing empty tile count

---

---

## Week 4: N-Tuple Networks and 2048 Learning Representation  
**Goal:** Understand how N-Tuple Networks work and how they are used to represent and learn state features in 2048.

### Topics Covered:
- **What are N-Tuple Networks?**
  - Local feature extractors for spatial games
  - Representation using pre-defined tile combinations
- **Why use N-Tuple Networks in 2048?**
  - Efficient function approximation in large state spaces
  - Fast lookup and updates
- **How are they trained?**
  - Using Temporal Difference Learning
  - Tuple selection, feature indexing
- **Research Papers Studied:**
  - Szubert & Jaśkowski (2014): *TD Learning of N-Tuple Networks*

---

##  Week 5: Monte Carlo & Policy Gradient Methods  
**Goal:** Learn how RL agents can learn directly from complete episodes or by following policy gradients instead of value functions.

### Topics Covered:
- **Monte Carlo Methods**
  - First-visit and Every-visit MC prediction
  - Learning from episodic returns
- **Policy Gradient Methods**
  - REINFORCE algorithm
  - On-policy gradient updates
  - Stochastic policy networks and action selection

### Resources:
- Monte Carlo Methods Explained: https://youtu.be/bpUszPiWM7o
- Policy Gradient Methods: https://youtu.be/5P7I-xPq8u8

---

##  Week 6: Multistage Temporal Difference Learning  
**Goal:** Study and implement the ideas from Yeh et al. (2016) for handling 2048-like games using stage-wise learning.

### Topics Covered:
- **Key Ideas from Yeh et al. (2016):**
  - Breaking down the game into **multiple stages**
  - Learning separate value functions for each stage
- **Advantages:**
  - Specialization of value functions improves stability
  - Works better as the board complexity increases
- **Implementation Plan:**
  - Define stage transitions based on board complexity (e.g., max tile or move count)
  - Train separate agents/weights for each stage

### Resources:
-  TDL 2048:https://github.com/moporgic/TDL2048
-  See paper: [Yeh et al. (2016) on IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/7518633)

---


## Week 7: Advanced TD Techniques (Jaśkowski, 2017)  
**Goal:** Implement techniques from Jaśkowski (2017) to improve 2048 agent performance.


### Resources:
- Paper: Jaśkowski, 2017 (arXiv)
- Code: https://github.com/alanhyue/RL-2048-with-n-tuple-network

---

## Week 8: N-Tuple Network Agent  
**Goal:** Build and optimize an N-Tuple based agent for 2048.

### Resources:
- https://github.com/Arnav3657/SoC-2048
