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
from board_game import BoardGame
from alphazero import main

class HexGame(BoardGame):
    def __init__(self, args):
        """
        Initialize Hex game environment
        
        Args:
            args: Configuration object containing:
                height (int): Number of rows in game board
                width (int): Number of columns in game board
        """
        self.height = args.height
        self.width = args.width
        self.input_shape = [3, self.height, self.width] 
        self.n_actions = self.width * self.width

    @property
    def name(self):
        """ 
        """
        return "MystRl Alpha Hex"

    @property   
    def grid_shape(self):
        """     
        """     
        return "hex"

    @property
    def place_mode(self):
        """
        """
        return "grid"

    @property
    def board_color(self):
        """
        """
        return (228, 200, 150)

    @property
    def line_color(self):
        """
        """
        return (0, 0, 0)

    @property
    def player_1_color(self):
        """
        """
        return (220, 60, 50)

    @property
    def player_2_color(self):
        """
        """
        return (30, 144, 255)

    def get_possible_action(self, game_state, pos):
        """
        """
        return pos

    def update_last_move(self, game_state, action):
        """
        """
        game_state.last_move = action

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

    def has_winning_path(self, state, player):
        """                                                                                             
        Performs a traversal to find a winning path for the given player.                               
        """                                                                                             
        visited = np.zeros_like(state, dtype=bool)                                                      
        stack = []                                                                                      
                                                                                                        
        # Define starting positions based on the player                                                 
        if player == 1:  # Top edge for Player 1                                                        
            for c in range(self.width):                                                                 
                if state[0, c] == player:                                                               
                    stack.append((0, c))                                                                
                    visited[0, c] = True                                                                
        elif player == -1:  # Left edge for Player 2                                                    
            for r in range(self.height):                                                                
                if state[r, 0] == player:                                                               
                    stack.append((r, 0))                                                                
                    visited[r, 0] = True                                                                
                                                                                                        
        # Perform iterative Depth-First Search                                                          
        while stack:                                                                                    
            r, c = stack.pop()                                                                          
                                                                                                        
            # Check for win condition                                                                   
            if player == 1 and r == self.height - 1:                                                    
                return True  # Reached the bottom edge                                                  
            if player == -1 and c == self.width - 1:                                                    
                return True  # Reached the right edge 

            # Explore neighbors                                                                         
            # A common way to represent hex grid neighbors on a square grid                             
            neighbors = [                                                                               
                (r - 1, c), (r + 1, c),                                                                 
                (r, c - 1), (r, c + 1),                                                                 
                (r - 1, c + 1), (r + 1, c - 1)                                                          
            ]                                                                                           
                                                                                                        
            for nr, nc in neighbors:                                                                    
                # Check if the neighbor is within bounds                                                
                if 0 <= nr < self.height and 0 <= nc < self.width:                                      
                    # Check if it's a valid, unvisited piece for the current player                     
                    if not visited[nr, nc] and state[nr, nc] == player:                                 
                        visited[nr, nc] = True                                                          
                        stack.append((nr, nc))                                                          
                                                                                                        
        return False # No winning path found 

    def check_winner(self, state):
        """
        Check win conditions for all possible lines
        
        Args:
            state (numpy.array): Game board to evaluate
            
        Returns:
            int: 0 = no winner, 1 = player1 wins, -1 = player2 wins
        """
        if self.has_winning_path(state, 1):                                                             
            return 1                                                                                    
        if self.has_winning_path(state, -1):                                                            
            return -1                                                                                   
                                                                                                        
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
        new_state = self.get_next_state(state, action, player) 

        winner = self.check_winner(new_state)
        done = (winner != 0)      
        if not winner:
            done = self.check_done(new_state, -player)        

        return new_state, winner, done
 
    def reset(self):
        """
        Initialize and return the starting board configuration for Reversi (Othello).
    
        Returns:
            ndarray: A 2D integer array of shape (height, width) with:
                     - Default dimensions defined by self.height and self.width
                     - Data type as 8-bit integers (int8)
        """ 
        return np.zeros([self.height, self.width], dtype=np.int8)


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
    main(HexGame, add_custom_argument)
   
