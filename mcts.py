#encoding=utf-8
#◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
# Author: Xiaoyuan Yao
# GitHub: https://github.com/yaoxiaoyuan/mystRL/
# Contact: yaoxiaoyuan1990@gmail.com
# Created: Sat Jun 14 15:08:00 2025
# License: MIT
# Version: 0.1.0
#
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
import numpy as np
import torch
from logger import logger

class Node: 
    def __init__(self, env, c_puct, parent, idx, state, player, winner, is_terminal):
        """
        Initialize a MCTS tree node with game state, player, and environment references.
        Stores game state information and initializes child node structures and statistics.
        """
        self.env = env
        self.c_puct = c_puct
        self.parent = parent
        self.idx = idx
        self.state = state
        self.player = player
        self.winner = winner
        self.is_terminal = is_terminal
        self.visits = 0
        self.children = []
        self.valid_moves = self.env.get_valid_moves(self.state, self.player)

    def is_leaf(self):
        """
        Determine if the current node is a leaf (unexpanded) node.
        Returns True if no children exist, indicating expansion is needed.
        """
        return len(self.children) == 0
        
    def expand(self, prior):
        """
        Expand the node by creating child nodes for all valid actions.
        Uses prior probabilities for action exploration and initializes visit statistics.
        Does nothing if the node represents a terminal game state.
        """
        if self.is_terminal:
            return

        self.prior = prior
        self.children_visits = np.zeros_like(self.valid_moves, dtype=float)
        self.children_value = np.zeros_like(self.valid_moves, dtype=float) 
        self.children_prior = self.prior[self.valid_moves]
        for idx,action in enumerate(self.valid_moves):
            new_state, winner, done = self.env.step(self.state, action, self.player)
            opponent = -self.player
            self.children.append(
                Node(self.env, self.c_puct, self, idx, new_state, opponent, winner, done)
            )
        
    def select(self):
        """
        Traverse the tree from this node using PUCT algorithm.
        Recursively selects child with highest UCT until reaching leaf node.
        Returns the leaf node where simulation/expansion will occur.
        """
        if self.is_leaf():
            return self

        q = self.children_value 

        u = self.c_puct * self.children_prior * np.sqrt(self.visits) / (1 + self.children_visits)

        uct = q + u

        idx = uct.argmax()
        return self.children[idx].select()

    def backup(self, value):
        """
        Propagate evaluation value backward through tree.
        Updates node statistics (visit counts and value estimates) recursively.
        Alternates value perspective when moving between players.
        """
        if self.parent:
            total_value = self.parent.children_value[self.idx] * self.parent.children_visits[self.idx] 
            total_value = total_value - value
            self.parent.visits += 1
            self.parent.children_visits[self.idx] += 1 
            self.parent.children_value[self.idx] = total_value / self.parent.children_visits[self.idx]
            self.parent.backup(-value) 

        
class MCTS:
    def __init__(self, env, n_envs, c_puct, model, n_simulations, device, states=None, players=None):
        """
        Initialize Monte Carlo Tree Search with parallel environments.
        
        Args:
            env: Game environment instance
            n_envs: Number of parallel environments
            c_puct: Exploration constant for UCT formula
            model: Neural network for policy/value predictions
            n_simulations: Max simulations per environment
            device: Compute device (CPU/GPU)
            states: Optional initial states for environments
            players: Optional current players for states
        """ 
        self.env = env
        self.n_envs = n_envs
        self.model = model
        self.n_simulations = n_simulations

        if states is None:
            self.root_list = [
                Node(env, c_puct, None, None, env.reset(), 1, 0, False)
                for _ in range(n_envs)
            ]
        else:
            assert n_envs == len(states) == len(players)
            self.root_list = [
                Node(env, c_puct, None, None, state, player, 0, False)
                for state,player in zip(states, players)
            ]

        self.device = device

    def is_done(self):
        """
        Check if all environments have terminated.
        
        Returns:
            bool: True if all environments have terminal states
        """ 
        return all(root.is_terminal for root in self.root_list)

    def simulate(self):
        """
        Execute Monte Carlo Tree Search simulations across all environments.
        Performs selection -> expansion -> neural evaluation -> backpropagation.
        """
        for i in range(self.n_simulations):

            nodes = []
            for root in self.root_list:
                if root.is_terminal:
                    continue
                if root.visits >= self.n_simulations:
                    continue
                nodes.append(root.select())

            if len(nodes) == 0:
                break

            model_inputs = np.stack(
                [
                    self.env.convert_to_model_inputs(node.state, node.player) 
                    for node in nodes
                ]
            )
            model_inputs = torch.tensor(model_inputs, dtype=torch.float).to(self.device)
            
            self.model.eval()    
            with torch.no_grad():
                logits, values = self.model(model_inputs)
                priors = torch.softmax(logits, -1)
                priors, values = priors.cpu().numpy(), values.cpu().numpy()
                
            for j,node in enumerate(nodes):
                node.expand(priors[j])
                if node.winner != 0:
                    real_value = (node.winner * node.player)
                    node.backup(real_value)
                else:
                    node.backup(values[j])
