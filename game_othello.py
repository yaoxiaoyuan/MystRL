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
import sys
import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from board_game import BoardGame
from alphazero import main

class OthelloGame(BoardGame):
    def __init__(self, args):
        """
        """
        self.height = args.height
        self.width = args.width
        self.input_shape = [3, self.height, self.width]
        self.n_actions = self.height * self.width + 1

    @property
    def name(self):
        """ 
        """
        return "MystRl Alpha Othello"

    @property
    def place_mode(self):
        """
        """
        return "grid"

    @property
    def board_color(self):
        """
        """
        return (143, 188, 143)

    @property
    def line_color(self):
        """
        """
        return (0, 0, 0)

    @property
    def player_1_color(self):
        """
        """
        return (0, 0, 0)

    @property
    def player_2_color(self):
        """
        """
        return (255, 255, 255)

    def get_possible_action(self, game_state, pos):
        """
        """
        return pos

    @property
    def can_pass_no_move(self):
        """
        """
        return True

    def update_last_move(self, game_state, action):
        """
        """
        game_state.last_move = action
        if action == self.width * self.height:
            game_state.last_move = -1

    def get_next_state(self, state, action, player):
        """
        """
        if action == self.height * self.width:
            return state.copy()

        row = action // self.width
        col = action % self.width
        new_state = state.copy()
        
        # Place player's disc at position
        new_state[row, col] = player
        
        # Directions for flipping checks (8 directions)
        directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        
        # Flip opponent discs in valid directions
        for dr, dc in directions:
            flip_list = []
            r, c = row + dr, col + dc
            
            # Collect opponent discs along direction
            while 0 <= r < self.height and 0 <= c < self.width and new_state[r, c] == -player:
                flip_list.append((r, c))
                r += dr
                c += dc
            
            # Validate flip if bounded by player's disc
            if 0 <= r < self.height and 0 <= c < self.width and new_state[r, c] == player:
                for (r_flip, c_flip) in flip_list:
                    new_state[r_flip, c_flip] = player
        
        return new_state

    def get_valid_moves(self, state, player):
        """
        """
        valid_actions = []
        for r in range(self.height):
            for c in range(self.width):
                if state[r, c] == 0 and self._is_valid_move(state, (r, c), player):
                    valid_actions.append(r * self.width + c)

        if len(valid_actions) == 0:
            valid_actions.append(self.height*self.width)

        return np.array(valid_actions)

    def no_valid_move(self, state, player):
        """
        """
        for r in range(self.height):
            for c in range(self.width):
                if state[r, c] == 0 and self._is_valid_move(state, (r, c), player):
                    return False
        return True

    def check_winner(self, state):
        """
        """
        # Count discs for final score
        count_1 = np.sum(state == 1)
        count_2 = np.sum(state == -1)
        
        if count_1 > count_2:
            return 1
        elif count_2 > count_1:
            return -1
        else:
            return 0  # Draw

    def check_done(self, state, player):
        """
        """
        # Board is completely filled
        if np.all(state != 0):
            return True
        
        # Check if both players have no valid moves
        return self.no_valid_move(state, 1) and self.no_valid_move(state, -1)

    def convert_to_model_inputs(self, state, player):
        """
        """
        return np.stack((
            state == 1,   
            state == -1,  
            np.full_like(state, player)  
        )).astype(np.float32)

    def step(self, state, action, player):
        """
        """
        new_state = self.get_next_state(state, action, player)
        winner = 0
        done = self.check_done(new_state, player)
        if done:
            winner = self.check_winner(new_state)
        return new_state, winner, done

    def reset(self):
        """
        """
        board = np.zeros((self.height, self.width), dtype=np.int8)
        # Set up center pieces
        mid_r, mid_c = self.height//2, self.width//2
        board[mid_r-1][mid_c-1] = -1 
        board[mid_r-1][mid_c] = 1    
        board[mid_r][mid_c-1] = 1    
        board[mid_r][mid_c] = -1     
        return board

    def _is_valid_move(self, state, position, player):
        """
        """
        r, c = position
        if state[r, c] != 0:
            return False
        
        # Check all 8 directions
        directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        for dr, dc in directions:
            r_temp, c_temp = r + dr, c + dc
            found_opponent = False
            
            # Traverse direction
            while (0 <= r_temp < self.height and 
                   0 <= c_temp < self.width and 
                   state[r_temp, c_temp] == -player):
                found_opponent = True
                r_temp += dr
                c_temp += dc
            
            # If valid sandwich structure found
            if (0 <= r_temp < self.height and 
                0 <= c_temp < self.width and 
                state[r_temp, c_temp] == player and 
                found_opponent):
                return True
                
        return False

def add_custom_argument(parser):
    """
    """
    parser.add_argument("--width",
                        type=int,
                        default=8,
                        help="game board width")

    parser.add_argument("--height",
                        type=int,
                        default=8,
                        help="game board height")

    return parser

if __name__ == "__main__":
    main(OthelloGame, add_custom_argument)
   
