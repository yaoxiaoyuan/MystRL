# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 23:24:52 2025

@author: 1
"""

try:
    import pygame
except:
    pass

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
        self.cell_size = 80
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
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(self.name)
    
    def render(self):
        """
        """
        pygame.draw.rect(self.screen, self.board_color, 
                         (self.padding, self.padding, self.board_width, self.board_height))
        if self.place_mode == "point":

            for row in range(self.height):
                start_pos = (self.padding+self.cell_size//2, self.padding + (row+0.5)*self.cell_size)
                end_pos = (self.padding - self.cell_size//2 + self.board_width, self.padding + (row+0.5)*self.cell_size)
                pygame.draw.line(self.screen, self.line_color, start_pos, end_pos, 1)

            for col in range(self.width):
                start_pos = (self.padding + (col+0.5)*self.cell_size, self.padding+self.cell_size//2)
                end_pos = (self.padding + (col+0.5)*self.cell_size, self.padding - self.cell_size//2 + self.board_height)
                pygame.draw.line(self.screen, self.line_color, start_pos, end_pos, 1)
        else:
            for row in range(self.height+1):
                start_pos = (self.padding+self.cell_size//2, self.padding + (row+0.5)*self.cell_size)
                end_pos = (self.padding - self.cell_size//2 + self.board_width, self.padding + (row+0.5)*self.cell_size)
                pygame.draw.line(self.screen, self.line_color, start_pos, end_pos, 1)

            for col in range(self.width+1):
                start_pos = (self.padding + (col+0.5)*self.cell_size, self.padding+self.cell_size//2)
                end_pos = (self.padding + (col+0.5)*self.cell_size, self.padding - self.cell_size//2 + self.board_height)
                pygame.draw.line(self.screen, self.line_color, start_pos, end_pos, 1)            
            
        pygame.display.flip()
        
if __name__ == "__main__":

    game = BoardGame()
    game.height = 9
    game.width = 9
    game.init_pygame()
    while True:
        game.render()
        input()

        
        
        
        
        