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

BACKGROUND = (20, 20, 35)
GAME_BG = (15, 15, 30)
INFO_BG = (30, 30, 50)
GRID_COLOR = (40, 40, 60)
FOOD_COLOR = (220, 60, 60)
HEAD_COLOR = (0, 150, 150)
TAIL_COLOR = (0, 200, 200)
TEXT_COLOR = (200, 220, 255)
ACCENT_COLOR = (70, 210, 210)
PANEL_BG = (35, 35, 55, 220)
HIGHLIGHT_COLOR = (250, 250, 255)

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class SnakeGame():
    """
    """
    def __init__(self, args):
        """
        """ 
        self.width = args.width
        self.height = args.height
        self.state_info = [
            {
                "shape": [args.n_last_frames+1, self.height, self.width],
                "dtype": "float"
            }
        ]

        self.n_actions = 4

        self.up = 0
        self.right = 1
        self.down = 2
        self.left = 3
        self.act2str = {
            self.up: "up", 
            self.right: "right",
            self.left: "left",
            self.down: "down"
        }

        self.direction_map = {
            self.up: (0, -1),
            self.right: (1, 0),
            self.down: (0, 1),
            self.left: (-1, 0),
        }

        self.highest = 0
        self.scores = []
        self.cnt = 0
        self.score = 0

        self.render_mode = args.render_mode
        if args.render_mode == "gui" and not GUI_RENDER_AVAILABLE :
            logger.warn("pygame not install, use text render mode.")
            args.render_mode = "text"

        if args.render_mode == "gui":
            self.fps = args.fps
            self.clock = pygame.time.Clock()
            
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            if not pygame.get_init():
                pygame.init()
                
            pygame.display.set_caption('MystRL Snake')
            
            self.font_score_title = pygame.font.SysFont('Arial', 24)
            self.font_score_value = pygame.font.SysFont('Arial', 32, bold=True)


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
        self.snake = [(self.height//2, self.width//2)]
        self.food = self.generate_food()   
        self.direction = 1
        self.state = self.create_new_state()
        self.update_state()
        self.time_since_last_food = 0 
        self.score = 0
        return self.state       
 

    def check_collision(self, position):
        """Check collision with walls or self"""
        x, y = position
        
        # Wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        
        # Self collision (except tail which is about to move)
        if position in self.snake[:-1]:
            return True
        
        return False


    def generate_food(self):
        """Generate food in an empty position"""
        # Find all empty positions
        empty_positions = [(x, y) for y in range(self.height) 
                          for x in range(self.width) 
                          if (x, y) not in self.snake]
        
        return random.choice(empty_positions)


    def update_state(self):
        """
        """
        new_state = self.create_new_state()
        new_state[0][:-2] = self.state[0][1:-1]
        max_value = 1
        min_value = 0.1
        delta_value = (max_value-min_value) / (self.height*self.width-1)
        for i,(x,y) in enumerate(self.snake):
            new_state[0][-2,y,x] = max_value - i*delta_value
        new_state[0][-1] = 0
        if self.food:
            new_state[0][-1,self.food[1],self.food[0]] = 1
        self.state = new_state


    def step(self, action):
        """
        """ 
        if action is not None and abs(action - self.direction) != 2:
            self.direction = action
        
        dx,dy = self.direction_map[self.direction]
        new_head = (self.snake[0][0]+dx, self.snake[0][1]+dy)

        dead_reward = 0
        win_reward = 0
        food_reward = 0

        max_len = self.width * self.height
        if self.check_collision(new_head):
            dead_reward = -15
            done = True
        elif new_head == self.food:
            self.snake.insert(0, new_head)
            if len(self.snake) == max_len:
                win_reward = 15
                done = True
                self.food = None
            else:
                food_reward = 1
                done = False
                self.food = self.generate_food()
            self.time_since_last_food = 0
            self.update_state()
            self.score += 1
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            done = False
            self.time_since_last_food += 1
            self.update_state()

        reward = (dead_reward + win_reward + food_reward)
        
        self.highest = max(self.highest, len(self.snake)-1)
        if done:
            self.scores.append(len(self.snake)-1)
            avg = sum(self.scores[-100:]) / min(100, len(self.scores))
            self.cnt += 1
            logger.info(f"{self.cnt} game done, score: {len(self.snake)-1}, avg: {avg}, highest:{self.highest}") 
         
        return self.state, reward, done 
        
    
    def process_input(self):
        """
        """
        if self.render_mode == "text":
            while True:
                action = input("0: up, 1: right, 2: down, 3:left\nInput action:")
                try:
                    action = int(action)
                    assert action in [0, 1, 2, 3]
                    return True, action
                except:
                    print("invalid action!")
        else:
            running, action = True, self.direction
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running, action = False,None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        running, action = True,self.up
                    elif event.key == pygame.K_DOWN:
                        running, action = True,self.down
                    elif event.key == pygame.K_LEFT:
                        running, action = True,self.left
                    elif event.key == pygame.K_RIGHT:
                        running, action = True,self.right
            
            if action is not None and abs(action - self.direction) != 2:
                self.direction = action
            
            return running, action
            
    def render(self):
        """
        """
        if self.render_mode == "text": 
            for i in range(self.height):
                for j in range(self.width):
                    if (j,i) in self.snake:
                        if (j,i) == self.snake[0]:
                            print("H ", end="")
                        else:
                            print("s ", end="")
                    elif (j,i) == self.food:
                        print("f ", end="")
                    else:
                        print("0 ", end="")
                print("")
            print(f"score:{self.score}") 
        else:
            self.screen.fill(BACKGROUND)
            
            #draw wall
            grid_size = int(min(0.65*WINDOW_HEIGHT/self.height, 0.65*WINDOW_WIDTH/self.width))
            
            board_width = grid_size*self.width
            board_height = grid_size*self.height
            
            top_y = (WINDOW_HEIGHT - board_height) / 2
            top_x = (0.7*WINDOW_WIDTH - board_width) / 2
            
            game_area = pygame.Rect(top_x, top_y, board_width, board_height)  
            pygame.draw.rect(self.screen, GAME_BG, game_area)
            
            for y in range(self.height+1):
                pygame.draw.line(self.screen, GRID_COLOR, (top_x, top_y+y*grid_size), (top_x+board_width, top_y+y*grid_size), 1)
            for x in range(self.width+1):
                pygame.draw.line(self.screen, GRID_COLOR, (top_x+x*grid_size, top_y), (top_x+x*grid_size, top_y+board_height), 1)
            
            #draw snake
            for i,(x,y) in enumerate(self.snake):
                ratio = i / (self.width*self.height)
                color = (
                    HEAD_COLOR[0] + (TAIL_COLOR[0] - HEAD_COLOR[0]) * ratio,
                    HEAD_COLOR[1] + (TAIL_COLOR[1] - HEAD_COLOR[1]) * ratio,
                    HEAD_COLOR[2] + (TAIL_COLOR[2] - HEAD_COLOR[2]) * ratio
                    )
        
    
                rect = pygame.Rect(
                    x * grid_size + top_x + 2,
                    y * grid_size + top_y + 2,
                    grid_size - 4,
                    grid_size - 4
                )
        
                pygame.draw.rect(self.screen, color, rect, border_radius=grid_size//4)
        
        
                if i == 0:
                    pygame.draw.circle(self.screen, (230, 250, 250), 
                                       (rect.centerx, rect.centery), 
                                       grid_size//5)
            
            if self.food is not None:
                food_radius = grid_size // 3
                pygame.draw.circle(self.screen, FOOD_COLOR, 
                                   (top_x + self.food[0] * grid_size + grid_size//2, 
                                    top_y + self.food[1] * grid_size + grid_size//2), 
                                   food_radius)
                pygame.draw.circle(self.screen, (255, 220, 220), 
                                   (top_x + self.food[0] * grid_size + grid_size//2, 
                                    top_y + self.food[1] * grid_size + grid_size//2), 
                                   food_radius // 2)
            
            panel_surface = pygame.Surface((0.26*WINDOW_WIDTH, WINDOW_HEIGHT-40), pygame.SRCALPHA)
            panel_surface.fill(PANEL_BG)
            self.screen.blit(panel_surface, (0.7*WINDOW_WIDTH, 20))
            
            avg = 0
            if len(self.scores) > 0:
                avg = sum(self.scores) // len(self.scores)
            infos = [
                ("Games Played", len(self.scores)),
                ("Current Score", self.score),
                ("Highest Score", self.highest),
                ("Average Score", avg)
            ]
            
            info_x = 0.73*WINDOW_WIDTH
            info_y_start = 40
            for i, (label, value) in enumerate(infos):
                y = info_y_start + i * 140
                
                label_text = self.font_score_title.render(label, True, TEXT_COLOR)
                self.screen.blit(label_text, (info_x, y))
                
                value_text = self.font_score_value.render(str(value), True, HIGHLIGHT_COLOR)
                self.screen.blit(value_text, (info_x, y + 40))
            
            pygame.display.update()                                                                      
            pygame.display.flip() 
   
        return True


class SnakeDQN(nn.Module):
    def __init__(self, args):
        super(SnakeDQN, self).__init__()
        n_actions, in_channels = 4, args.n_last_frames+1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 7, padding=3),                                                          
            nn.ReLU(inplace=True),                                                                      
        )                                                                                               
        self.pool = nn.AdaptiveAvgPool2d((4, 4))                                                        
        self.feature_size = 128 * 4 * 4                                                                 
                                                                                                        
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
                        default=15,
                        help="game board width")

    parser.add_argument("--height",
                        type=int,
                        default=8,
                        help="game board height")

    parser.add_argument("--use_dueling_dqn",
                        default=False,
                        action='store_true',
                        help="if true, use dueling dqn")
    
    return parser
 

if __name__ == "__main__":

    main(SnakeGame, SnakeDQN, add_custom_argument)
     
