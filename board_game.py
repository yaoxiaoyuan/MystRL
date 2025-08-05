# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 23:24:52 2025

@author: 1
"""

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

class Button:
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
        self.font = pygame.font.SysFont('Arial', 30)
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
        return "point"

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
    
    def init_pygame(self):
        """
        """
        pygame.init()
        
        self.padding = 5
        self.cell_size = 35
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
        self.reset_button = Button(
            (self.window_width + self.board_width - self.reset_button_width) / 2,
            self.window_height - 50, 
            self.reset_button_width, 
            40,
            "Reset",
            self.window_width,
            self.window_height)
        
        self.font = pygame.font.SysFont('Arial', 32)
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(self.name)
        self.clock = pygame.time.Clock()

    def render(self, state, player, winner, done):
        """
        """
        self.screen.fill(BACKGROUND)
 
        pygame.draw.rect(self.screen, self.board_color, 
                         (self.padding, self.padding, self.board_width, self.board_height))

        for row in range(self.height+1):
            start_pos = (self.padding+self.cell_size//2, self.padding + (row+0.5)*self.cell_size)
            end_pos = (self.padding - self.cell_size//2 + self.board_width, self.padding + (row+0.5)*self.cell_size)
            pygame.draw.line(self.screen, self.line_color, start_pos, end_pos, 1)

        for col in range(self.width+1):
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
        
        self.reset_button.draw(self.screen)
   
        line_height = self.font.get_linesize()
        status_text_x = 2*self.padding + self.board_width + 10
        status_text_y = 20
        
        if done:
            text = "Game Over!"
            text_surface = self.font.render(text, True, ACCENT_COLOR)
            self.screen.blit(text_surface, (status_text_x, status_text_y))
            
            if winner == 0:
                text = "It's a draw!"
                text_surface = self.font.render(text, True, ACCENT_COLOR)
                self.screen.blit(text_surface, (status_text_x, status_text_y + line_height))
            else:
                text = "Winner:"
                text_surface = self.font.render(text, True, ACCENT_COLOR)
                self.screen.blit(text_surface, (status_text_x, status_text_y + line_height))
                
                color = self.player_2_color 
                if winner == -1:
                    color = self.player_1_color
                
                text_width = text_surface.get_width()
                circle_x = status_text_x + text_width + line_height
                circle_y = status_text_y + line_height + line_height // 2
            
                pygame.draw.circle(self.screen, color, (circle_x, circle_y), line_height//2)
            
        else:
            text = "Current Player:"
            text_surface = self.font.render(text, True, ACCENT_COLOR)
            self.screen.blit(text_surface, (status_text_x, status_text_y))
            color = self.player_2_color 
            if player == 1:
                color = self.player_1_color
            
            text_width = text_surface.get_width()
            circle_x = status_text_x + text_width + line_height
            circle_y = status_text_y + line_height // 2
            
            pygame.draw.circle(self.screen, color, (circle_x, circle_y), line_height//2)

            
        pygame.display.flip()
        self.clock.tick(60)
       
    def process_input(self):
        """
        """
        running, action, reset, human = True, None, False, None
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
                   action = row * self.width + col 
            reset,human = self.reset_button.handle_event(event)

        return running, action, reset, human
        
if __name__ == "__main__":
    import numpy as np
    game = BoardGame()
    game.height = 8
    game.width = 8
    state = np.zeros([game.height+1, game.width+1], dtype=np.int8)
    game.init_pygame()
    player = 1
    running = True
    while running:
        game.render(state, player, 0, False)
        running, action, reset, human = game.process_input()
        if reset:
           state = np.zeros([game.height+1, game.width+1], dtype=np.int8)
        if action is not None:
            state[action//game.width, action%game.width] = player
            player = -player
            print(state) 
        
        
        
