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
import random
import numpy as np
import torch.nn as nn
GUI_RENDER_AVAILABLE=False
try:
    import pygame
    GUI_RENDER_AVAILABLE=True
except:
    pass

from dqn import main
from logger import logger

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BOARD_COLS = 10
BOARD_ROWS = 20
BLOCK_SIZE = 30
BOARD_PADDING = 20
INFO_WIDTH = 250
INFO_PADDING = 20

BACKGROUND = (15, 15, 30)
BOARD_BG = (25, 25, 40)
GRID_COLOR = (45, 45, 65)
BLOCK_COLOR = (200, 200, 255)
GHOST_COLOR = (100, 100, 150, 150)
CURRENT_BLOCK_COLOR = (0, 200, 255)
NEXT_BLOCK_COLOR = (0, 200, 180)
TEXT_COLOR = (230, 230, 255)
HIGHLIGHT_COLOR = (250, 250, 255)
BORDER_COLOR = (90, 90, 160)
PANEL_BG = (35, 35, 55, 220)

class TetrisGame:
    def __init__(self, args):

        self.n_last_frames = args.n_last_frames
        self.height = args.height
        self.width = args.width
        self.n_actions = 5
        
        self.state_info = [                                                                                      
                {                                                                                                      
                    "shape": [self.n_last_frames+1, self.height, self.width],
                    "dtype": "float"
                },
            ]

        self.shapes = [
            np.array([[1, 1, 1, 1]], dtype=int),  # I
            np.array([[1, 1], [1, 1]], dtype=int),  # O
            np.array([[0, 1, 0], [1, 1, 1]], dtype=int),  # T
            np.array([[0, 1, 1], [1, 1, 0]], dtype=int),  # S
            np.array([[1, 1, 0], [0, 1, 1]], dtype=int),  # Z
            np.array([[1, 0, 0], [1, 1, 1]], dtype=int),  # L
            np.array([[0, 0, 1], [1, 1, 1]], dtype=int)  # J
         ]

        self.up = 0                                                                                     
        self.right = 1                                                                                  
        self.down = 2                                                                                   
        self.left = 3                                                                                   
        self.space = 4
        
        self.act2str = {
            self.up: "rotate",
            self.right: "right",
            self.left: "left",
            self.down: "down",
            self.space: "hard_drop"
        }
 
        self.board = np.zeros([self.height, self.width], dtype=int)
        self.score = 0
        self.cleared = 0
        self.done = False
        self.next_piece = None
        self.reset()
        self.cnt = 0
        
        self.highest = 0
        self.scores = []
        self.cleared_list = []
        self.fall_speed = 2
        self.fall_time = 0

        self.fps = args.fps
        self.render_mode = args.render_mode
        if args.render_mode == "gui":
            self.fps = args.fps
            self.clock = pygame.time.Clock()
            
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            if not pygame.get_init():
                pygame.init()
            
            pygame.display.set_caption("MystRL Tetris")
            
            self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
            self.font_medium = pygame.font.SysFont("Arial", 26, bold=True)
            self.font_small = pygame.font.SysFont("Arial", 20)
            self.font_info = pygame.font.SysFont("Courier", 18)
            
    def create_new_state(self):
        """
        """
        state = []
        for info in self.state_info:
            dtype = np.int32 if info['dtype'] == 'int' else np.float32
            state.append(np.zeros(
                info['shape'], dtype=dtype
            ))
        return state


    def reset(self):
        """
        """
        self.board = np.zeros([self.height, self.width], dtype=int)
        self.new_piece() 
        self.game_over = False
        self.score = 0
        self.cleared = 0
        self.state = self.create_new_state()
        self.update_state()
        return self.state  
 
    def merge_piece(self):
        for i in range(len(self.current_piece)):                                                        
            for j in range(len(self.current_piece[i])):                                                 
                if self.current_piece[i][j]:                                                            
                    self.board[self.current_piece_y+i][self.current_piece_x+j] = 1     


    def clear_lines(self):
        lines_to_clear = []
        reward = 0
        for i in range(self.height):
            if self.board[i].all():
                lines_to_clear.append(i)
                self.cleared += 1
        for line in lines_to_clear:
            self.board = np.delete(self.board, line, axis=0)
            self.board = np.insert(self.board, 0, 0, axis=0)

        score = [0, 100, 300, 500, 800][len(lines_to_clear)]
        self.score += score
        reward += (score/100)
        return reward    
          
    def update_state(self):
        new_state = self.create_new_state()
        new_state[0][:-2] = self.state[0][1:-1]
        new_state[0][-2] = self.board
        new_state[0][-2] = new_state[0][-2] / 2
        for i in range(len(self.current_piece)):
            for j in range(len(self.current_piece[i])):
                if self.current_piece[i][j]:
                    new_state[0][-2][self.current_piece_y+i, self.current_piece_x+j] = 1
         
        dx = (self.width - len(self.next_piece[0])) // 2 
        dy = (self.height - len(self.next_piece)) // 2
        for i in range(len(self.next_piece)):
            for j in range(len(self.next_piece[i])):
                if self.next_piece[i][j]:
                    new_state[0][-1][dy+i, dx+j] = 1        
        self.state = new_state
 
    def rotate(self, clockwise=True):
        if clockwise:
            rotated = np.rot90(self.current_piece, -1)
        else:
            rotated = np.rot90(self.current_piece, 1)

        if not self.check_collision(rotated, self.current_piece_x, self.current_piece_y):
            self.current_piece = rotated
 
    def move(self, dx, dy):
        move_x = self.current_piece_x + dx
        move_y = self.current_piece_y + dy
        if not self.check_collision(self.current_piece, move_x, move_y):
            self.current_piece_x = move_x
            self.current_piece_y = move_y
            return True
        return False


    def hard_drop(self):
        while self.move(0, 1):
            pass


    def check_collision(self, piece, piece_x, piece_y):    
        for i in range(len(piece)):
            for j in range(len(piece[i])):
                x = piece_x + j
                y = piece_y + i
                if x < 0 or x >= self.width:
                    return True
                if y < 0 or y >= self.height:
                    return True
                if piece[i][j] and self.board[y][x]:
                    return True
        return False

    def new_piece(self):
        if self.next_piece is None:
            self.next_piece = random.choice(self.shapes)
        self.current_piece = self.next_piece
        self.current_piece_x = (self.width - len(self.current_piece[0])) // 2                           
        self.current_piece_y = 0                                                                        
        if self.check_collision(self.current_piece, self.current_piece_x, self.current_piece_y):
            return False
        self.next_piece = random.choice(self.shapes) 
        return True
 
    def calculate_score(self):
        """
        """
        height_weight = 0.5
        hole_weight = 0.36
        bumpiness_weight = 0.18

        col_max = self.board.argmax(axis=0)
        has_block = np.any(self.board, axis=0)
        heights = np.where(has_block, self.height - col_max, 0)
        aggregate_height = np.sum(heights)
        height_score = aggregate_height 

        accum = np.cumsum(self.board, axis=0)
        total_holes = np.sum((self.board == 0) & (accum > 0))
        hole_score = total_holes        
 
        bumpiness = np.sum(np.abs(np.diff(heights)))
        bumpiness_score = bumpiness        

        #print(f"height:{heights}, hole:{total_holes}, bumpiness:{bumpiness}")
        score = -(height_weight * height_score + 
                  hole_weight * hole_score +
                  bumpiness_weight * bumpiness_score)
        
        return score


    def step(self, action):

        old = self.calculate_score()

        if action == self.up: 
            self.rotate()     
        elif action == self.right:
            self.move(1, 0)
        elif action == self.down:        
            self.move(0, 1)
        elif action == self.left:
            self.move(-1, 0) 
        elif action == self.space:
            self.hard_drop()

        done = False
        reward = 0
        self.fall_time += 1
        if self.fall_time >= self.fall_speed:
            self.fall_time = 0
            if not self.move(0, 1):
                self.merge_piece()
                reward = reward + self.clear_lines()
                success = self.new_piece()
                if not success:
                    reward = reward - 100
                    done = True
                    self.cnt += 1
                    self.scores.append(self.score)
                    self.cleared_list.append(self.cleared)
                    
                    avg = sum(self.scores[-100:]) / min(100, len(self.scores))
                    avg_cleared = sum(self.cleared_list[-100:]) / min(100, len(self.scores))
                    logger.info(f"{self.cnt} game done, score: {self.score}, cleared: {self.cleared} avg: {avg}, avg_cleared: {avg_cleared} highest: {self.highest}")
        
        self.highest = max(self.highest, self.score)
        reward = reward + self.calculate_score() - old

        self.update_state()

        return self.state, reward, done

 
    def draw_block(self, x, y, color, size=BLOCK_SIZE, is_current=False):
        """
        """
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, color, rect)
        
        if is_current:

            pygame.draw.rect(self.screen, (min(255, color[0] + 80), min(255, color[1] + 80), min(255, color[2] + 80)), 
                             pygame.Rect(x, y, size, 4))
            pygame.draw.rect(self.screen, (min(255, color[0] + 80), min(255, color[1] + 80), min(255, color[2] + 80)), 
                             pygame.Rect(x, y, 4, size))

            pygame.draw.rect(self.screen, (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60)), 
                             pygame.Rect(x + size - 4, y, 4, size))
            pygame.draw.rect(self.screen, (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60)), 
                             pygame.Rect(x, y + size - 4, size, 4))
        else:

            pygame.draw.rect(self.screen, (min(255, color[0] + 30), min(255, color[1] + 30), min(255, color[2] + 30)), 
                             pygame.Rect(x, y, size, 4))
            pygame.draw.rect(self.screen, (min(255, color[0] + 30), min(255, color[1] + 30), min(255, color[2] + 30)), 
                             pygame.Rect(x, y, 4, size))
            pygame.draw.rect(self.screen, (max(0, color[0] - 30), max(0, color[1] - 30), max(0, color[2] - 30)), 
                             pygame.Rect(x + size - 4, y, 4, size))
            pygame.draw.rect(self.screen, (max(0, color[0] - 30), max(0, color[1] - 30), max(0, color[2] - 30)), 
                             pygame.Rect(x, y + size - 4, size, 4))


    def process_input(self):
        """
        """
        if self.render_mode == "text":
            while True:
                action = input("0: up, 1: right, 2: down, 3:left, 4:space\nInput action:")
                try:
                    action = int(action)
                    assert action in [0, 1, 2, 3, 4]
                    return True, action
                except:
                    print("invalid action!")
        else:    
            running, action = True, None                                                                                
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running, action = False, None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        running, action = True,self.up
                    elif event.key == pygame.K_DOWN:
                        running, action = True,self.down
                    elif event.key == pygame.K_LEFT:
                        running, action = True,self.left
                    elif event.key == pygame.K_RIGHT:
                        running, action = True,self.right
                    elif event.key == pygame.K_SPACE:
                        running, action = True,self.space
            return running, action
                    

    def render(self):
        if self.render_mode == "text": 
            board = [[int(self.board[i][j]) for j in range(self.width)] for i in range(self.height)]  
            for i in range(len(self.current_piece)):
                for j in range(len(self.current_piece[i])):
                    if self.current_piece[i][j]:
                        board[self.current_piece_y+i][self.current_piece_x+j] = 2
            for line in board:
                print(line)
            print(f"next:\n{self.next_piece}")
            print(f"score:{self.score}")
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
            
            panel_surface = pygame.Surface((INFO_WIDTH, WINDOW_HEIGHT - 2*BOARD_PADDING), pygame.SRCALPHA)
            panel_surface.fill(PANEL_BG)
            
            block_size = int(min(0.8*WINDOW_HEIGHT/(self.height), 0.8*WINDOW_WIDTH/self.width))
            board_width = self.width * block_size
            board_height = self.height * block_size
            
            board_x = (WINDOW_WIDTH - 2*BOARD_PADDING - INFO_WIDTH - board_width) // 2
            board_y = (WINDOW_HEIGHT - board_height) / 2
            
            pygame.draw.rect(self.screen, BOARD_BG, (board_x - 5, board_y - 5, board_width + 10, board_height + 10))
            pygame.draw.rect(self.screen, BORDER_COLOR, (board_x - 5, board_y - 5, board_width + 10, board_height + 10), 3)
            
            for row in range(self.height):
                for col in range(self.width):
                    if row < len(self.board) and col < len(self.board[0]) and self.board[row][col] == 1:
                        x = board_x + col * block_size
                        y = board_y + row * block_size
                        self.draw_block(x, y, BLOCK_COLOR, size=block_size)
            
                piece_rows, piece_cols = self.current_piece.shape
                for r in range(piece_rows):
                    for c in range(piece_cols):
                        if self.current_piece[r][c] == 1:
            
                            board_r = self.current_piece_y + r
                            board_c = self.current_piece_x + c
                            
                            if 0 <= board_r < self.height and 0 <= board_c < self.width:
                                x = board_x + board_c * block_size
                                y = board_y + board_r * block_size
                                self.draw_block(x, y, CURRENT_BLOCK_COLOR, size=block_size, is_current=True)

            info_x = board_x + board_width + BOARD_PADDING * 2
            info_y = 20
            self.screen.blit(panel_surface, (info_x, info_y))
            
            next_text = self.font_medium.render("Next Piece", True, TEXT_COLOR)
            self.screen.blit(next_text, (info_x + (INFO_WIDTH - next_text.get_width()) // 2, info_y + 30))
            
            if self.next_piece is not None:
                preview_rows, preview_cols = self.next_piece.shape
                preview_size = min(BLOCK_SIZE, BLOCK_SIZE * 3 // max(preview_rows, preview_cols))
                preview_width = preview_cols * preview_size
                
                preview_x = info_x + (INFO_WIDTH - preview_width) // 2
                preview_y = info_y + 80
                
                for r in range(preview_rows):
                    for c in range(preview_cols):
                        if self.next_piece[r][c] == 1:
                            self.draw_block(preview_x + c * preview_size, 
                                            preview_y + r * preview_size, 
                                            NEXT_BLOCK_COLOR,
                                            size=preview_size)
            
            avg = 0
            if len(self.scores) > 0:
                avg = sum(self.scores) // len(self.scores)
                
            info_y_start = info_y + 170
            infos = [
                ("Games Played", len(self.scores)),
                ("Current Score", self.score),
                ("Highest Score", self.highest),
                ("Average Score", avg)
            ]
            
            for i, (label, value) in enumerate(infos):
                y = info_y_start + i * 50
                
                label_text = self.font_small.render(label, True, TEXT_COLOR)
                self.screen.blit(label_text, (info_x + 30, y))
                
                value_text = self.font_medium.render(str(value), True, HIGHLIGHT_COLOR)
                self.screen.blit(value_text, (info_x + INFO_WIDTH - 30 - value_text.get_width(), y - 3))
            
            controls_y = info_y + 380
            controls = [
                "CONTROLS:",
                "← → : Move",
                "↑ : Rotate",
                "↓ : Soft Drop",
                "SPACE : Hard Drop",
            ]
            
            for i, text in enumerate(controls):
                ctrl_text = self.font_info.render(text, True, (180, 220, 255) if i == 0 else (150, 200, 230))
                self.screen.blit(ctrl_text, (info_x + (INFO_WIDTH - ctrl_text.get_width()) // 2, controls_y + i * 30))
            
            pygame.display.update()                                                                      
            pygame.display.flip()

            clock = pygame.time.Clock()
            clock.tick(self.fps) 

        return True


class TetrisDQN(nn.Module):
    def __init__(self, args):
        super(TetrisDQN, self).__init__()
        n_actions, in_channels = 5, args.n_last_frames+1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 7, padding=3),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))                                                        
        self.feature_size = 256 * 4 * 4                                                                 
                                                                                                        
        self.feature_compressor = nn.Sequential(                                                        
            nn.Linear(self.feature_size, 256),                                                          
            nn.ReLU(inplace=True)                                                                       
        )                                                                                               
                                                                                                        
        self.use_dueling_dqn = args.use_dueling_dqn                                                     
        if args.use_dueling_dqn:                                                                        
            self.value_stream = nn.Linear(256, 1)                                                           
            self.advantage_stream = nn.Linear(256, n_actions)                                           
        else:                                                                                           
            self.fc = nn.Linear(256, n_actions)  

    def forward(self, state):
        x = state[0]
 
        conv_out = self.conv_layers(x)
        pooled = self.pool(conv_out) 
        flat = pooled.view(pooled.size(0), -1)

        features = self.feature_compressor(flat)

        if self.use_dueling_dqn:
            values = self.value_stream(features) 
            advantages = self.advantage_stream(features)

            return values + advantages - advantages.mean(1, keepdim=True)
        return self.fc(features)


def add_custom_argument(parser):
    """
    Adds custom command-line arguments to an argument parser.

    This function extends an existing ArgumentParser object with game configuration
    and model selection options. 
 
    Parameters:
    parser (argparse.ArgumentParser): The argument parser to enhance

    Returns:
    argparse.ArgumentParser: The modified parser with added arguments
    """
    parser.add_argument("--width",
                        type=int,
                        default=10,
                        help="game board width")

    parser.add_argument("--height",
                        type=int,
                        default=20,
                        help="game board height")

    parser.add_argument("--use_dueling_dqn",
                        default=False, 
                        action='store_true',
                        help="if true, use dueling dqn")
    
    return parser
                                                    

if __name__ == "__main__":
    main(TetrisGame, TetrisDQN, add_custom_argument)

