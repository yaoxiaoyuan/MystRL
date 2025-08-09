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
from dataclasses import dataclass
import numpy as np
try:
    import pygame
    import pygame.gfxdraw
except:
    pass

BACKGROUND = (240, 236, 225)
BUTTON_COLOR = (100, 100, 120)
BUTTON_HOVER = (130, 130, 150) 
BUTTON_ACTIVE = (80, 80, 100)  
TEXT_COLOR = (250, 250, 250)  
OPTION_BUTTON = (90, 140, 170) 
OPTION_HOVER = (110, 160, 190) 
ACCENT_COLOR = (180, 70, 60)   
LAST_FOCUS_COLOR = (0, 100, 255)

class Button:
    def __init__(self, x, y, width, height, text, window_width, window_height):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = BUTTON_COLOR
        self.hover_color = BUTTON_HOVER
        self.active_color = BUTTON_ACTIVE
        self.is_hovered = False
        self.is_active = False
        self.font = pygame.font.SysFont('Arial', 32)
        self.small_font = pygame.font.SysFont('Arial', 16)
        self.window_width = window_width
        self.window_height = window_height   
        
    def draw(self, surface):
        if self.is_active:
            color = self.hover_color if self.is_hovered else self.color
            pygame.draw.rect(surface, color, self.rect, border_radius=8)
            pygame.draw.rect(surface, ACCENT_COLOR, self.rect, 2, border_radius=8)
            
            text_surf = self.font.render(self.text, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=self.rect.center)
            surface.blit(text_surf, text_rect)
        
    def handle_event(self, event):

        if self.is_active:

            if event.type == pygame.MOUSEMOTION:
                self.is_hovered = self.rect.collidepoint(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.rect.collidepoint(event.pos):
                        return True
        
        return False

class ResetButton:
    def __init__(self, x, y, width, height, text, window_width, window_height):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = BUTTON_COLOR
        self.hover_color = BUTTON_HOVER
        self.active_color = BUTTON_ACTIVE
        self.is_hovered = False
        self.is_active = False
        self.show_options = False
        self.options = ["First Player", "Second Player"]
        self.option_buttons = []
        self.create_option_buttons()
        self.font = pygame.font.SysFont('Arial', 32)
        self.small_font = pygame.font.SysFont('Arial', 16)
        self.window_width = window_width
        self.window_height = window_height   
        
    def create_option_buttons(self):
        self.option_buttons = []
        for i, option in enumerate(self.options):
            btn_rect = pygame.Rect(
                self.rect.x + self.rect.width // 2 - 10,
                self.rect.y - 95 + i * 50,
                140, 
                40
            )
            self.option_buttons.append((btn_rect, option))
    
    def draw(self, surface):
        color = self.active_color if self.is_active else (self.hover_color if self.is_hovered else self.color)
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, ACCENT_COLOR, self.rect, 2, border_radius=8)
        
        text_surf = self.font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
        if self.show_options:
            overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150)) 
            surface.blit(overlay, (0, 0))
            
            options_rect = pygame.Rect(
                self.rect.x + self.rect.width // 2 - 60,
                self.rect.y - 135,
                200,
                140,
            )
            pygame.draw.rect(surface, BACKGROUND, options_rect, border_radius=10)
            pygame.draw.rect(surface, BUTTON_COLOR, options_rect, 3, border_radius=10)
            
            title = self.small_font.render("Select Starting Player", True, BUTTON_COLOR)
            title_rect = title.get_rect(center=(options_rect.centerx, options_rect.y + 25))
            surface.blit(title, title_rect)
            
            for btn_rect, option_text in self.option_buttons:
                opt_color = OPTION_HOVER if btn_rect.collidepoint(pygame.mouse.get_pos()) else OPTION_BUTTON
                pygame.draw.rect(surface, opt_color, btn_rect, border_radius=6)
                pygame.draw.rect(surface, ACCENT_COLOR, btn_rect, 2, border_radius=6)
                
                text_surf = self.small_font.render(option_text, True, TEXT_COLOR)
                text_rect = text_surf.get_rect(center=btn_rect.center)
                surface.blit(text_surf, text_rect)
    
    def handle_event(self, event):

        reset = False
        human = None
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.rect.collidepoint(event.pos) and not self.show_options:
                    self.is_active = True
                    self.show_options = True
                    return reset, human
                elif self.show_options:
                    for i,(btn_rect, option_text) in enumerate(self.option_buttons):
                        if btn_rect.collidepoint(event.pos):
                            self.is_active = False
                            self.show_options = False
                            reset = True
                            human = (-1)**i
                    if not any(btn_rect.collidepoint(event.pos) for btn_rect, _ in self.option_buttons):
                        self.is_active = False
                        self.show_options = False
                        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.is_active:
                self.is_active = False
        
        return reset, human


@dataclass
class GameState:
    state: np.array
    render_mode: str = "gui"
    turn: int = 0
    human: int = 1
    player: int = 0
    winner: int = 0
    done: bool = True
    last_move: int = -1
    select: int = 0
    thinking: bool = False

    def reset(self, state, human):
        self.turn = 0
        self.state = state
        self.human = human
        self.player = 1
        self.winner = 0
        self.done = False
        self.last_move = -1


class BoardGame():
    
    @property
    def name(self):
        """
        """
        return "Board Game"
    
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
        return (0,0,0)
    
    @property
    def player_2_color(self):
        """
        """    
        return (255,255,255)
    
    @property
    def can_pass_no_move(self):
        """
        """
        return False

    def init_pygame(self):
        """
        """
        pygame.init()
        
        self.padding = 5
        self.cell_size = 40
        self.info_width = 300
        
        if self.place_mode == "grid":
            self.board_height = (self.height+1) * self.cell_size
            self.board_width = (self.width+1) * self.cell_size
            self.window_height = 3*self.padding + self.board_height
            self.window_width = 2*self.padding + self.board_width + self.info_width
        else:
            self.board_height = self.height * self.cell_size
            self.board_width = self.width * self.cell_size
            self.window_height = 3*self.padding + self.board_height 
            self.window_width = 2*self.padding + self.board_width + self.info_width
         
        self.radius = int(self.cell_size * 0.4)

        self.reset_button_width = self.info_width * 2 // 3
        self.reset_button = ResetButton(
            (self.window_width + self.board_width - self.reset_button_width) / 2,
            self.window_height - 50, 
            self.reset_button_width, 
            40,
            "New Game",
            self.window_width,
            self.window_height)
        
        self.pass_button = Button(
            (self.window_width + self.board_width - self.reset_button_width) / 2,
            self.window_height - 100,  
            self.reset_button_width,
            40,
            "Pass",
            self.window_width,
            self.window_height)
 
        self.font = pygame.font.SysFont('Arial', 24)
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(self.name)
        self.clock = pygame.time.Clock()

    def render_cli(self, game_state):
        """
        """
        if game_state.player == 0:
            return
        print("Current board:")
        if not game_state.done:
            print(f"Current player: {game_state.player}")
        else:
            if game_state.winner != 0:
                print(f"Game over! Winner: {game_state.winner}")
            else:
                print("Game over! It's a draw")
        print("-" * 20) 

    def draw_focus_box(self, row, col, color):
        """
        """         
        if self.place_mode == "point":
            center_x = self.padding + (col+0.5)*self.cell_size
            center_y = self.padding + (row+0.5)*self.cell_size
        else:  
            center_x = self.padding + (col+1)*self.cell_size
            center_y = self.padding + (row+1)*self.cell_size

        start_pos = (center_x - 0.45*self.cell_size, center_y - 0.45*self.cell_size)
        end_pos = (center_x - 0.35*self.cell_size, center_y - 0.45*self.cell_size)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        start_pos = (center_x - 0.45*self.cell_size, center_y - 0.45*self.cell_size)
        end_pos = (center_x - 0.45*self.cell_size, center_y - 0.35*self.cell_size)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        start_pos = (center_x - 0.45*self.cell_size, center_y + 0.45*self.cell_size)
        end_pos = (center_x - 0.35*self.cell_size, center_y + 0.45*self.cell_size)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        start_pos = (center_x - 0.45*self.cell_size, center_y + 0.45*self.cell_size)
        end_pos = (center_x - 0.45*self.cell_size, center_y + 0.35*self.cell_size)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        start_pos = (center_x + 0.45*self.cell_size, center_y - 0.45*self.cell_size)
        end_pos = (center_x + 0.35*self.cell_size, center_y - 0.45*self.cell_size)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        start_pos = (center_x + 0.45*self.cell_size, center_y - 0.45*self.cell_size)
        end_pos = (center_x + 0.45*self.cell_size, center_y - 0.35*self.cell_size)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        start_pos = (center_x + 0.45*self.cell_size, center_y + 0.45*self.cell_size)
        end_pos = (center_x + 0.35*self.cell_size, center_y + 0.45*self.cell_size)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 1)

        start_pos = (center_x + 0.45*self.cell_size, center_y + 0.45*self.cell_size)
        end_pos = (center_x + 0.45*self.cell_size, center_y + 0.35*self.cell_size)
        pygame.draw.line(self.screen, color, start_pos, end_pos, 2)

    def render(self, game_state):
        """
        """
        if game_state.render_mode == "text":
            self.render_cli(game_state)
            return
 
        state = game_state.state
        player = game_state.player
        winner = game_state.winner
        done = game_state.done
        last_move = game_state.last_move
        turn = game_state.turn
        human = game_state.human
        thinking = game_state.thinking

        self.screen.fill(BACKGROUND)
 
        pygame.draw.rect(self.screen, self.board_color, 
                         (self.padding, self.padding, self.board_width, self.board_height))

        height,width = self.height, self.width
        if self.place_mode == "point":
            height,width = height-1,width-1
 
        for row in range(height+1):
            start_pos = (self.padding+self.cell_size//2, self.padding + (row+0.5)*self.cell_size)
            end_pos = (self.padding - self.cell_size//2 + self.board_width, self.padding + (row+0.5)*self.cell_size)
            pygame.draw.line(self.screen, self.line_color, start_pos, end_pos, 1)

        for col in range(width+1):
            start_pos = (self.padding + (col+0.5)*self.cell_size, self.padding+self.cell_size//2)
            end_pos = (self.padding + (col+0.5)*self.cell_size, self.padding - self.cell_size//2 + self.board_height)
            pygame.draw.line(self.screen, self.line_color, start_pos, end_pos, 1)

        for i in range(self.height):
            for j in range(self.width):
                if state[i][j] == 0:
                    continue

                color = self.player_2_color
                if state[i][j] == 1:
                    color = self.player_1_color

                if self.place_mode == "point":
                    center_x = self.padding + (j+0.5)*self.cell_size 
                    center_y = self.padding + (i+0.5)*self.cell_size 
                else: 
                    center_x = self.padding + (j+1)*self.cell_size  
                    center_y = self.padding + (i+1)*self.cell_size 
                pygame.draw.circle(
                                  self.screen,
                                  color, 
                                  (center_x, center_y),
                                  self.radius)

        if player in [1, -1]:

            line_height = self.font.get_linesize()
            status_text_x = 2*self.padding + self.board_width + 10
            status_text_y = 20
            
            if done:
                status_text = "Game Over!"
                text_surface = self.font.render(status_text, True, ACCENT_COLOR)
                self.screen.blit(text_surface, (status_text_x, status_text_y))
                 
                status_text = "It's a draw!"
                if winner == human:
                    status_text = "Winner: Human!"
                elif winner == -human:
                    status_text = "Winner: Alpha!"
                text_surface = self.font.render(status_text, True, ACCENT_COLOR)
                self.screen.blit(text_surface, (status_text_x, status_text_y + line_height))
                
                if winner == human or winner == -human:
                    color = self.player_1_color
                    if winner == -1:
                        color = self.player_2_color
                    text_width = text_surface.get_width()
                    circle_x = status_text_x + text_width + line_height * 2 // 3
                    circle_y = status_text_y + line_height + line_height // 2
                    pygame.draw.circle(self.screen, color, (circle_x, circle_y), line_height//2)

            else:

                self.pass_button.is_active = False
                if self.can_pass_no_move:
                    if player == human and self.no_valid_move(state, player):
                        self.pass_button.is_active = True
 
                for i,(status_text, role) in enumerate([
                        ["Human:", human],
                        ["AI:", -human],
                        [f"Current Turn: {turn+1}", None],
                        ["Current Player:" + (" Human" if player == human else "AI"), player]
                    ]):
                    
                    text_surface = self.font.render(status_text, True, ACCENT_COLOR)
                    self.screen.blit(text_surface, (status_text_x, i*line_height + status_text_y))
                    
                    if role is None:
                        continue

                    color = self.player_1_color
                    if role == -1:
                        color = self.player_2_color
                    text_width = text_surface.get_width()
                    circle_x = status_text_x + text_width + line_height * 2 // 3
                    circle_y = status_text_y + i*line_height + line_height // 2
                    pygame.draw.circle(self.screen, color, (circle_x, circle_y), line_height//2)

                if thinking:
                    status_text = "Thinking..."
                    text_surface = self.font.render(status_text, True, ACCENT_COLOR)
                    self.screen.blit(text_surface, (status_text_x, status_text_y + (i+1)*line_height))

                if last_move >= 0:
                    row, col = last_move // self.width, last_move % self.width
                    self.draw_focus_box(row, col, LAST_FOCUS_COLOR)

        self.reset_button.draw(self.screen)
        self.pass_button.draw(self.screen)
 
        pygame.display.flip()
        self.clock.tick(60)
       
    def process_input(self, game_state):
        """
        """
        if game_state.render_mode == "text":
            if game_state.player != 0 and game_state.player != game_state.human:
                return True, False, None, None
            try:
                user_input = int(input("-2:exit, -1:reset, other:action of current game\ninput:"))
                if user_input == -2:
                    return False, False, None, None
                elif user_input == -1:
                    user_input = int(input("1:first, -1:second\ninput:"))
                    if user_input == 1:
                        return True, True, 1, None
                    if user_input == -1:
                        return True, True, -1, None
                else:
                    return True, False, None, user_input
            except:
                pass
            return True, False, None, None

        running, pos, reset, human = True, None, False, None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
               running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
               x, y = event.pos
               if self.place_mode == "point":
                   row = (y - self.padding) // self.cell_size
                   col = (x - self.padding) // self.cell_size
               else:
                   row = (y - self.padding - self.cell_size//2) // self.cell_size
                   col = (x - self.padding - self.cell_size//2) // self.cell_size              
               if row >= 0 and row <= self.height and col >= 0 and col <= self.width:
                   pos = row * self.width + col 
            reset,human = self.reset_button.handle_event(event)
            click_pass = self.pass_button.handle_event(event)
            if click_pass:
                pos = self.width * self.height

        return running, reset, human, pos
        
