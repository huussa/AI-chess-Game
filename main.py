import queue
import threading

import pygame

from chess_ai import best_move, load_tables, train_both_colours
from chess_engine import BLACK, WHITE, Board, Move, opponent

WINDOW = 720
SQUARE = WINDOW // 8
LIGHT, DARK = (240, 217, 181), (181, 136, 99)
SELECTED, TARGET = (246, 246, 105), (106, 168, 79)
PIECES = {"wK":"♔", "wQ":"♕", "wR":"♖", "wB":"♗", "wN":"♘", "wP":"♙", "bK":"♚", "bQ":"♛", "bR":"♜", "bB":"♝", "bN":"♞", "bP":"♟"}

MENU_BG_TOP, MENU_BG_BOTTOM = (36, 29, 47), (82, 54, 68)
MENU_CARD, MENU_BORDER = (47, 39, 59), (211, 170, 112)
PLAY, PLAY_HOVER, QUIT, QUIT_HOVER = (198, 143, 72), (225, 170, 91), (105, 70, 83), (138, 88, 105)
WHITE_CHOICE, WHITE_CHOICE_HOVER = (188, 169, 146), (219, 201, 177)


def board_to_view(square, view_color):
    """Convert a logical board square to its displayed position."""
    row, col = square
    return (row, col) if view_color == WHITE else (7 - row, 7 - col)


def view_to_board(square, view_color):
    """Convert a clicked display square back to logical board coordinates."""
    row, col = square
    return (row, col) if view_color == WHITE else (7 - row, 7 - col)


def draw(screen, board, selected, legal, font, small, status, view_color):
    for view_row in range(8):
        for view_col in range(8):
            row, col = view_to_board((view_row, view_col), view_color)
            rect = pygame.Rect(view_col * SQUARE, view_row * SQUARE, SQUARE, SQUARE)
            pygame.draw.rect(screen, LIGHT if (row + col) % 2 == 0 else DARK, rect)
            if selected == (row, col): pygame.draw.rect(screen, SELECTED, rect, 6)
            if any(m.end == (row, col) for m in legal):
                pygame.draw.circle(screen, TARGET, rect.center, 12)
            piece = board.grid[row][col]
            if piece:
                image = font.render(PIECES[piece], True, (248,248,248) if piece[0] == WHITE else (28,28,28))
                screen.blit(image, image.get_rect(center=rect.center))
    for i in range(8):
        file_label = "abcdefgh"[i if view_color == WHITE else 7 - i]
        rank_label = str(8 - i if view_color == WHITE else i + 1)
        screen.blit(small.render(file_label, True, (60, 45, 30)), (i*SQUARE+4, WINDOW-20))
        screen.blit(small.render(rank_label, True, (60, 45, 30)), (4, i*SQUARE+2))
    text = small.render(status, True, (30,30,30)); screen.blit(text, (10, 8))
    pygame.display.flip()


def rounded_button(screen, rect, label, font, base, hover, mouse):
    active = rect.collidepoint(mouse)
    color = hover if active else base
    shadow = rect.move(0, 6)
    pygame.draw.rect(screen, (24, 18, 29), shadow, border_radius=18)
    pygame.draw.rect(screen, color, rect, border_radius=18)
    pygame.draw.rect(screen, (242, 218, 177), rect, width=2, border_radius=18)
    image = font.render(label, True, (255, 250, 238))
    screen.blit(image, image.get_rect(center=rect.center))
    return active


def draw_chess_background(screen):
    """The original start-menu backdrop, shared by all menu screens."""
    for y in range(WINDOW):
        amount = y / WINDOW
        color = tuple(int(MENU_BG_TOP[i] * (1 - amount) + MENU_BG_BOTTOM[i] * amount) for i in range(3))
        pygame.draw.line(screen, color, (0, y), (WINDOW, y))
    # Keep the light, subtle board pattern from the first menu design.
    for row in range(9):
        for col in range(9):
            if (row + col) % 2 == 0:
                pygame.draw.rect(screen, (255, 242, 215, 12), (col * 90 - 45, row * 90 - 45, 90, 90))


def draw_menu(screen, fonts):
    title_font, subtitle_font, button_font, icon_font = fonts
    draw_chess_background(screen)

    card = pygame.Rect(110, 92, 500, 545)
    pygame.draw.rect(screen, (20, 16, 29), card.move(0, 9), border_radius=30)
    pygame.draw.rect(screen, MENU_CARD, card, border_radius=30)
    pygame.draw.rect(screen, MENU_BORDER, card, width=2, border_radius=30)

    crown = icon_font.render("♛", True, (226, 186, 119))
    screen.blit(crown, crown.get_rect(center=(WINDOW // 2, 172)))
    title = title_font.render("CHESS", True, (255, 242, 215))
    screen.blit(title, title.get_rect(center=(WINDOW // 2, 252)))
    subtitle = subtitle_font.render("Pygame  •  Q-learning foundation", True, (211, 190, 171))
    screen.blit(subtitle, subtitle.get_rect(center=(WINDOW // 2, 295)))
    pygame.draw.line(screen, MENU_BORDER, (242, 330), (478, 330), width=2)

    mouse = pygame.mouse.get_pos()
    play_rect = pygame.Rect(220, 380, 280, 72)
    quit_rect = pygame.Rect(220, 474, 280, 72)
    rounded_button(screen, play_rect, "PLAY", button_font, PLAY, PLAY_HOVER, mouse)
    rounded_button(screen, quit_rect, "QUIT", button_font, QUIT, QUIT_HOVER, mouse)
    hint = subtitle_font.render("Choose an option to begin", True, (189, 164, 167))
    screen.blit(hint, hint.get_rect(center=(WINDOW // 2, 588)))
    return play_rect, quit_rect


def draw_opponent_select(screen, fonts):
    title_font, subtitle_font, button_font, icon_font = fonts
    heading_font = pygame.font.SysFont("georgia", 39, bold=True)
    draw_chess_background(screen)

    card = pygame.Rect(90, 105, 540, 510)
    pygame.draw.rect(screen, (20, 16, 29), card.move(0, 9), border_radius=30)
    pygame.draw.rect(screen, MENU_CARD, card, border_radius=30)
    pygame.draw.rect(screen, MENU_BORDER, card, width=2, border_radius=30)
    icon = icon_font.render("♞", True, (226, 186, 119))
    screen.blit(icon, icon.get_rect(center=(WINDOW // 2, 171)))
    heading = heading_font.render("CHOOSE OPPONENT", True, (255, 242, 215))
    screen.blit(heading, heading.get_rect(center=(WINDOW // 2, 245)))
    message = subtitle_font.render("Who would you like to play against?", True, (211, 190, 171))
    screen.blit(message, message.get_rect(center=(WINDOW // 2, 288)))
    pygame.draw.line(screen, MENU_BORDER, (220, 321), (500, 321), width=2)

    mouse = pygame.mouse.get_pos()
    ai_rect = pygame.Rect(170, 365, 380, 72)
    player_rect = pygame.Rect(170, 462, 380, 72)
    rounded_button(screen, ai_rect, "PLAY AGAINST AI", button_font, PLAY, PLAY_HOVER, mouse)
    rounded_button(screen, player_rect, "PLAY AGAINST PLAYER", button_font, QUIT, QUIT_HOVER, mouse)
    hint = subtitle_font.render("Press Esc to return", True, (189, 164, 167))
    screen.blit(hint, hint.get_rect(center=(WINDOW // 2, 569)))
    return ai_rect, player_rect


def draw_color_select(screen, fonts, chosen_color):
    """Color picker shown after an opponent has been selected."""
    title_font, subtitle_font, button_font, icon_font = fonts
    heading_font = pygame.font.SysFont("georgia", 42, bold=True)
    draw_chess_background(screen)

    card = pygame.Rect(90, 82, 540, 556)
    pygame.draw.rect(screen, (20, 16, 29), card.move(0, 9), border_radius=30)
    pygame.draw.rect(screen, MENU_CARD, card, border_radius=30)
    pygame.draw.rect(screen, MENU_BORDER, card, width=2, border_radius=30)
    icon = icon_font.render("♔", True, (226, 186, 119))
    screen.blit(icon, icon.get_rect(center=(WINDOW // 2, 148)))
    heading = heading_font.render("CHOOSE COLOR", True, (255, 242, 215))
    screen.blit(heading, heading.get_rect(center=(WINDOW // 2, 219)))
    message = subtitle_font.render("Select your side, then begin the match.", True, (211, 190, 171))
    screen.blit(message, message.get_rect(center=(WINDOW // 2, 261)))
    pygame.draw.line(screen, MENU_BORDER, (220, 293), (500, 293), width=2)

    mouse = pygame.mouse.get_pos()
    white_rect = pygame.Rect(190, 322, 340, 64)
    black_rect = pygame.Rect(190, 401, 340, 64)
    start_rect = pygame.Rect(190, 497, 340, 72)
    rounded_button(screen, white_rect, "WHITE", button_font, WHITE_CHOICE, WHITE_CHOICE_HOVER, mouse)
    rounded_button(screen, black_rect, "BLACK", button_font, QUIT, QUIT_HOVER, mouse)
    selected_rect = white_rect if chosen_color == WHITE else black_rect
    pygame.draw.rect(screen, (255, 238, 151), selected_rect.inflate(8, 8), width=4, border_radius=21)
    rounded_button(screen, start_rect, "START", button_font, PLAY, PLAY_HOVER, mouse)
    selected = "White" if chosen_color == WHITE else "Black"
    hint = subtitle_font.render("Selected: " + selected + "    •    Esc to return", True, (189, 164, 167))
    screen.blit(hint, hint.get_rect(center=(WINDOW // 2, 600)))
    return white_rect, black_rect, start_rect


def draw_ai_training(screen, fonts, episodes, input_active, note):
    """AI setup UI only. Training behavior is intentionally added later."""
    title_font, subtitle_font, button_font, icon_font = fonts
    heading_font = pygame.font.SysFont("georgia", 38, bold=True)
    compact_button_font = pygame.font.SysFont("georgia", 20, bold=True)
    draw_chess_background(screen)

    card = pygame.Rect(70, 58, 580, 604)
    pygame.draw.rect(screen, (20, 16, 29), card.move(0, 9), border_radius=30)
    pygame.draw.rect(screen, MENU_CARD, card, border_radius=30)
    pygame.draw.rect(screen, MENU_BORDER, card, width=2, border_radius=30)
    icon = icon_font.render("♞", True, (226, 186, 119))
    screen.blit(icon, icon.get_rect(center=(WINDOW // 2, 119)))
    heading = heading_font.render("Q-LEARNING", True, (255, 242, 215))
    screen.blit(heading, heading.get_rect(center=(WINDOW // 2, 181)))
    subtitle = subtitle_font.render("Configure training before starting a game", True, (211, 190, 171))
    screen.blit(subtitle, subtitle.get_rect(center=(WINDOW // 2, 218)))
    pygame.draw.line(screen, MENU_BORDER, (190, 246), (530, 246), width=2)

    input_label = subtitle_font.render("Training episodes", True, (229, 210, 185))
    screen.blit(input_label, (128, 276))
    input_rect = pygame.Rect(128, 303, 210, 56)
    pygame.draw.rect(screen, (32, 27, 40), input_rect, border_radius=12)
    pygame.draw.rect(screen, (255, 238, 151) if input_active else MENU_BORDER, input_rect, width=3, border_radius=12)
    field_value = episodes or "Enter a number"
    field_color = (255, 242, 215) if episodes else (166, 148, 151)
    field_font = button_font if episodes else subtitle_font
    count_text = field_font.render(field_value, True, field_color)
    screen.blit(count_text, count_text.get_rect(center=input_rect.center))
    if input_active:
        cursor_x = input_rect.centerx + (count_text.get_width() // 2 if episodes else 0) + 3
        pygame.draw.line(screen, (255, 238, 151), (cursor_x, input_rect.top + 14), (cursor_x, input_rect.bottom - 14), width=2)

    mouse = pygame.mouse.get_pos()
    train_rect = pygame.Rect(358, 303, 234, 56)
    scratch_rect = pygame.Rect(128, 379, 464, 62)
    start_rect = pygame.Rect(128, 523, 290, 68)
    back_rect = pygame.Rect(438, 523, 154, 68)
    rounded_button(screen, train_rect, "TRAIN MODEL", compact_button_font, PLAY, PLAY_HOVER, mouse)
    rounded_button(screen, scratch_rect, "TRAIN FROM SCRATCH", button_font, QUIT, QUIT_HOVER, mouse)
    rounded_button(screen, start_rect, "START A GAME", button_font, PLAY, PLAY_HOVER, mouse)
    rounded_button(screen, back_rect, "BACK", button_font, QUIT, QUIT_HOVER, mouse)
    instruction = "Click the box and type the number of training episodes."
    instruction_text = subtitle_font.render(instruction, True, (189, 164, 167))
    screen.blit(instruction_text, instruction_text.get_rect(center=(WINDOW // 2, 468)))
    note_text = subtitle_font.render(note, True, (189, 164, 167))
    screen.blit(note_text, note_text.get_rect(center=(WINDOW // 2, 492)))
    return input_rect, train_rect, scratch_rect, start_rect, back_rect


def run_game(screen, clock, fonts, chosen_color, opponent_mode):
    font, small = pygame.font.SysFont("segoeuisymbol", 67), pygame.font.SysFont("arial", 20, bold=True)
    board, selected, legal = Board(), None, []
    tables = load_tables() if opponent_mode == "ai" else None
    computer_color = opponent(chosen_color) if opponent_mode == "ai" else None
    next_computer_move_at = pygame.time.get_ticks() + 450
    random_source = __import__("random").Random()
    while True:
        if board.outcome():
            status = board.outcome()
        elif board.in_check(board.turn):
            status = ("White" if board.turn == WHITE else "Black") + " is in check — escape, capture, or block"
        else:
            status = "White to move" if board.turn == WHITE else "Black to move"
        player_name = "White" if chosen_color == WHITE else "Black"
        status += "   |   Your color: " + player_name
        if opponent_mode == "ai":
            computer_name = "Black" if chosen_color == WHITE else "White"
            status += "   |   Computer: " + computer_name
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return "menu"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                board.reset(); selected, legal = None, []
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not board.outcome() and board.turn == chosen_color:
                view_square = (event.pos[1] // SQUARE, event.pos[0] // SQUARE)
                square = view_to_board(view_square, chosen_color)
                choices = [m for m in legal if m.end == square]
                if choices:
                    # Promotion defaults to queen for a quick, playable UI.
                    board.push(next((m for m in choices if m.promotion == "Q"), choices[0]))
                    selected, legal = None, []
                elif (piece := board.piece_at(square)) and piece[0] == board.turn:
                    selected = square; legal = [m for m in board.legal_moves() if m.start == square]
                else: selected, legal = None, []
        if (
            opponent_mode == "ai"
            and not board.outcome()
            and board.turn == computer_color
            and pygame.time.get_ticks() >= next_computer_move_at
        ):
            board.push(best_move(board, computer_color, tables, random_source))
            selected, legal = None, []
            next_computer_move_at = pygame.time.get_ticks() + 450
        draw(screen, board, selected, legal, font, small, status + "   [R] restart", chosen_color)
        clock.tick(60)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW, WINDOW))
    pygame.display.set_caption("Chess — Pygame + Q-learning base")
    fonts = (
        pygame.font.SysFont("georgia", 61, bold=True),
        pygame.font.SysFont("arial", 18),
        pygame.font.SysFont("georgia", 27, bold=True),
        pygame.font.SysFont("segoeuisymbol", 78),
    )
    clock, state = pygame.time.Clock(), "menu"
    chosen_color, opponent_mode = WHITE, "player"
    episodes, input_active = "", False
    training_note = "Training is configured here; it has not started."
    training_updates = queue.Queue()
    training_thread = None

    def start_training(reset):
        nonlocal training_thread, training_note
        if training_thread and training_thread.is_alive():
            training_note = "Training is already running."
            return
        try:
            episode_count = int(episodes)
            if episode_count <= 0:
                raise ValueError
        except ValueError:
            training_note = "Enter a whole number greater than zero."
            return

        def report(color, current, total):
            training_updates.put(("progress", color, current, total))

        def worker():
            try:
                train_both_colours(episode_count, reset=reset, progress=report)
                training_updates.put(("complete", episode_count))
            except Exception as error:
                training_updates.put(("error", str(error)))

        training_note = "Preparing training..."
        training_thread = threading.Thread(target=worker, daemon=True)
        training_thread.start()
    while True:
        if state == "menu":
            play_rect, quit_rect = draw_menu(screen, fonts)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if quit_rect.collidepoint(event.pos):
                        pygame.quit(); return
                    if play_rect.collidepoint(event.pos):
                        state = "opponent_select"
            pygame.display.flip()
        elif state == "opponent_select":
            ai_rect, player_rect = draw_opponent_select(screen, fonts)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    state = "menu"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if ai_rect.collidepoint(event.pos):
                        opponent_mode = "ai"
                        state = "ai_training"
                    elif player_rect.collidepoint(event.pos):
                        opponent_mode = "player"
                        state = "color_select"
            # Both pages share one Pygame window; refresh it after drawing the
            # selection view so the state change is visible immediately.
            pygame.display.flip()
        elif state == "color_select":
            white_rect, black_rect, start_rect = draw_color_select(screen, fonts, chosen_color)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    state = "opponent_select"
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if white_rect.collidepoint(event.pos):
                        chosen_color = WHITE
                    elif black_rect.collidepoint(event.pos):
                        chosen_color = BLACK
                    elif start_rect.collidepoint(event.pos):
                        result = run_game(screen, clock, fonts, chosen_color, opponent_mode)
                        if result == "quit":
                            pygame.quit(); return
                        state = "menu"
            pygame.display.flip()
        elif state == "ai_training":
            while not training_updates.empty():
                update = training_updates.get_nowait()
                if update[0] == "progress":
                    training_note = f"Training {update[1]}: {update[2]:,} / {update[3]:,}"
                elif update[0] == "complete":
                    training_note = f"Done — trained {update[1]:,} episodes for each color."
                else:
                    training_note = "Training error: " + update[1]
            input_rect, train_rect, scratch_rect, start_rect, back_rect = draw_ai_training(
                screen, fonts, episodes, input_active, training_note
            )
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        input_active = False
                        state = "opponent_select"
                    elif input_active and event.key == pygame.K_BACKSPACE:
                        episodes = episodes[:-1]
                    elif input_active and event.unicode.isdigit() and len(episodes) < 7:
                        episodes += event.unicode
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    input_active = input_rect.collidepoint(event.pos)
                    if train_rect.collidepoint(event.pos):
                        start_training(reset=False)
                    elif scratch_rect.collidepoint(event.pos):
                        start_training(reset=True)
                    elif start_rect.collidepoint(event.pos):
                        if training_thread and training_thread.is_alive():
                            training_note = "Wait for training to finish before starting a game."
                        else:
                            input_active = False
                            state = "color_select"
                    elif back_rect.collidepoint(event.pos):
                        input_active = False
                        state = "opponent_select"
            pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
