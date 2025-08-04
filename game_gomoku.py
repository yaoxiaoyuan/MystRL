import sys
import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
from alphazero import main

class GomokuGame:
    def __init__(self, args):
        """
        Initialize Connect4 game environment
        
        Args:
            args: Configuration object containing:
                height (int): Number of rows in game board
                width (int): Number of columns in game board
        """
        self.height = args.height
        self.width = args.width
        self.input_shape = [3, self.height, self.width] 
        self.n_actions = self.width * self.width

    def get_next_state(self, state, action, player):
        """
        Generate next game state after player's move
        
        Args:
            state (numpy.array): Current game board (height x width)
            action (int): Column index where piece is dropped
            player (int): Player ID (1 or -1)
            
        Returns:
            numpy.array: Updated game board after move
        """
        new_state = np.copy(state)

        row = action // self.width
        col = action % self.width 
        new_state[row, col] = player

        return new_state

    def get_valid_moves(self, state, player):
        """
        Identify legal moves at current state
        
        Args:
            state (numpy.array): Current game board
            player (int): Player ID (unused, preserved for interface consistency)
            
        Returns:
            numpy.array: Array of columns with available space
        """ 
        row,col = np.where(state==0) 
        valid_moves = row * self.width + col
        return valid_moves

    def check_winner(self, state):
        """
        Check win conditions for all possible lines
        
        Args:
            state (numpy.array): Game board to evaluate
            
        Returns:
            int: 0 = no winner, 1 = player1 wins, -1 = player2 wins
        """
        for i in range(self.height):
            for j in range(self.width):
                color = state[i, j]
                if color == 0:
                    continue
                    
                if j <= self.width - 5 and np.all(state[i, j:j+5] == color):
                    return color
                    
                if i <= self.height - 5 and np.all(state[i:i+5, j] == color):
                    return color
                     
                if i <= self.height - 5 and j <= self.width - 5:
                    diag = [state[i+k, j+k] for k in range(5)]
                    if np.all(diag == color):
                        return color
                        
                if i <= self.height - 5 and j >= 5:
                    anti_diag = [state[i+k, j-k] for k in range(5)]
                    if np.all(anti_diag == color):
                        return color
                        
        return 0  

    def check_done(self, state, player):
        """
        Check if board is completely filled (draw condition)
        
        Args:
            state (numpy.array): Game board to check
            player (int): Player ID (1 or -1)  
  
        Returns:
            bool: True if board full, False otherwise
        """
        return (state != 0).all()   

    def convert_to_model_inputs(self, state, player):
        """
        Convert game state to neural network input format
        
        Creates 3 channels:
        Channel 0: Positions of player 1's pieces
        Channel 1: Positions of player 2's pieces
        Channel 2: Constant value indicating current player
        
        Args:
            state (numpy.array): Original game board
            player (int): Current player ID (1 or -1)
            
        Returns:
            numpy.array: Shaped [3, height, width] for model input
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

        winner = self.check_winner(new_state)
        done = (winner != 0)      
        if not winner:
            done = self.check_done(new_state, -player)        

        return new_state, winner, done
 
    def reset(self):
        """
        Execute single game step (move + state update)
        
        Args:
            state (numpy.array): Current game board
            action (int): Selected column index
            player (int): Player making the move
            
        Returns:
            tuple: (new_state, winner, done)
                new_state: Updated game board
                winner: 0=no win, 1/-1=winning player
                done: Terminal state flag
        """
        return np.zeros([self.height, self.width], dtype=np.int8)

    def render(self, state, player, winner, done):
        """
        Initialize new game state
        
        Returns:
            numpy.array: Zero-initialized game board
        """
        print("Current board:")                                                                          
        print(state)                                                                                     
        if not done:                                                                                     
            print(f"Current player: {player}")                                                           
        else:                                                                                            
            if winner != 0:                                                                              
                print(f"Game over! Winner: {winner}")                                                    
            else:                                                                                        
                print("Game over! It's a draw")                                                          
        print("-" * 20)   

def add_custom_argument(parser):
    """
    """
    parser.add_argument("--width",
                        type=int,
                        default=9,
                        help="game board width")

    parser.add_argument("--height",
                        type=int,
                        default=9,
                        help="game board height")

    return parser


if __name__ == "__main__":
    main(GomokuGame, add_custom_argument)
   
