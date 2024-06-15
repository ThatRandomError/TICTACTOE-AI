import sys

import pygame
import pyn

nn = pyn.NN([1,1])

nn.load("ttt.pyn")

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 15
BOARD_SIZE = 3
SQUARE_SIZE = WIDTH // BOARD_SIZE
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

# Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')
screen.fill(BG_COLOR)

# Board (1D list)
board = [0] * (BOARD_SIZE * BOARD_SIZE)

# Functions
def draw_lines():
    # Horizontal lines
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH)
    # Vertical lines
    pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

def draw_figures():
    for i in range(BOARD_SIZE * BOARD_SIZE):
        row = i // BOARD_SIZE
        col = i % BOARD_SIZE
        if board[i] == 1:
            pygame.draw.line(screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH)
            pygame.draw.line(screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH)
        elif board[i] == 2:
            pygame.draw.circle(screen, CIRCLE_COLOR, (int(col * SQUARE_SIZE + SQUARE_SIZE // 2), int(row * SQUARE_SIZE + SQUARE_SIZE // 2)), CIRCLE_RADIUS, CIRCLE_WIDTH)

def mark_square(index, player):
    board[index] = player

def available_square(index):
    return board[index] == 0

def restart():
    screen.fill(BG_COLOR)
    draw_lines()
    for i in range(BOARD_SIZE * BOARD_SIZE):
        board[i] = 0

def update_display():
    screen.fill(BG_COLOR)
    draw_lines()
    draw_figures()

# Main loop
draw_lines()
player = 1  # X starts first
game_over = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
            mouseX = event.pos[0]  # x
            mouseY = event.pos[1]  # y

            clicked_row = mouseY // SQUARE_SIZE
            clicked_col = mouseX // SQUARE_SIZE
            index = clicked_row * BOARD_SIZE + clicked_col

            if available_square(index):
                mark_square(index, player)
                player = player % 2 + 1

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                restart()
                player = 1
                game_over = False
        
        if player == 1:
            nn.prediction(board)
            output = nn.output
            allow_output = False
            
            while allow_output == False:
                high = 0
                pos = 0
                for index, i in enumerate(output):
                    if i > high:
                        pos = index
                        high = i
                if board[pos] != 0:
                    output[pos] = 0
                else:
                    allow_output = True
            
            board[pos] = 1
            player = 2

    update_display()
    pygame.display.update()
