import queue
import threading

import pygame

from chess_ai import QTableStore, best_move, record_game_transition, state_id, train_both_colours
from chess_engine import BLACK, WHITE, Board, Move, opponent

WINDOW = 720
SQUARE = WINDOW // 8
LIGHT, DARK = (240, 217, 181), (181, 136, 99)
SELECTED, TARGET = (246, 246, 105), (106, 168, 79)
PIECES = {"wK":"♔", "wQ":"♕", "wR":"♖", "wB":"♗", "wN":"♘", "wP":"♙", "bK":"♚", "bQ":"♛", "bR":"♜", "bB":"♝", "bN":"♞", "bP":"♟"}

# Menus deliberately remain compact; the match gets its own, roomier layout.
GAME_WIDTH, GAME_HEIGHT = 1000, 860
GAME_BOARD_SIZE, GAME_SQUARE = 600, 75
GAME_BOARD_LEFT, GAME_BOARD_TOP = 46, 142
GAME_BOARD_RECT = pygame.Rect(GAME_BOARD_LEFT, GAME_BOARD_TOP, GAME_BOARD_SIZE, GAME_BOARD_SIZE)
LAST_MOVE = (247, 217, 87)
# The match screen uses the same plum, gold, and cream palette as the menus.
GAME_BG, GAME_PANEL, GAME_PANEL_DARK = (36, 29, 47), (47, 39, 59), (39, 32, 50)
GAME_BORDER, GAME_TEXT, GAME_MUTED = (211, 170, 112), (255, 242, 215), (211, 190, 171)
PIECE_POINTS = {"Q": 9, "R": 5, "B": 3, "N": 3, "P": 1}

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


def replay_board(moves, move_count):
    """Return a read-only historical position without changing the live game."""
    replay = Board()
    for move in moves[:move_count]:
        replay.push(move)
    return replay


def material_points(board, color):
    """Return the standard material value of the pieces a side still has."""
    return sum(
        PIECE_POINTS.get(piece[1], 0)
        for row in board.grid
        for piece in row
        if piece and piece[0] == color
    )


def material_difference(board, color):
    """Show this side's material lead or deficit against its opponent."""
    return material_points(board, color) - material_points(board, opponent(color))


def game_button(screen, rect, label, font, mouse, *, disabled=False, danger=False):
    hovered = rect.collidepoint(mouse) and not disabled
    if disabled:
        fill, edge, text_color = (61, 51, 68), (104, 85, 93), (156, 140, 145)
    elif danger:
        fill, edge, text_color = ((138, 88, 105) if hovered else (105, 70, 83)), GAME_BORDER, GAME_TEXT
    else:
        fill, edge, text_color = ((225, 170, 91) if hovered else (198, 143, 72)), GAME_BORDER, GAME_TEXT
    pygame.draw.rect(screen, (20, 16, 29), rect.move(0, 4), border_radius=13)
    pygame.draw.rect(screen, fill, rect, border_radius=13)
    pygame.draw.rect(screen, edge, rect, width=2, border_radius=13)
    image = font.render(label, True, text_color)
    screen.blit(image, image.get_rect(center=rect.center))


def player_panel(screen, rect, name, color, points, font, small):
    pygame.draw.rect(screen, GAME_PANEL, rect, border_radius=17)
    pygame.draw.rect(screen, GAME_BORDER, rect, width=2, border_radius=17)
    avatar = pygame.Rect(rect.x + 16, rect.y + 13, 48, 48)
    pygame.draw.circle(screen, (232, 218, 193) if color == WHITE else (76, 62, 79), avatar.center, 24)
    pygame.draw.circle(screen, GAME_BORDER, avatar.center, 24, width=2)
    accent = (91, 71, 81) if color == WHITE else (232, 218, 193)
    pygame.draw.circle(screen, accent, (avatar.centerx, avatar.y + 19), 8)
    pygame.draw.arc(screen, accent, (avatar.x + 10, avatar.y + 23, 28, 23), 180, 360, width=5)
    screen.blit(font.render(name, True, GAME_TEXT), (rect.x + 78, rect.y + 13))
    score = small.render(f"{points:+d} POINTS", True, GAME_MUTED)
    screen.blit(score, (rect.x + 80, rect.y + 42))


def draw_game_board(screen, board, selected, legal, piece_font, small, view_color, last_move):
    for view_row in range(8):
        for view_col in range(8):
            row, col = view_to_board((view_row, view_col), view_color)
            rect = pygame.Rect(
                GAME_BOARD_LEFT + view_col * GAME_SQUARE,
                GAME_BOARD_TOP + view_row * GAME_SQUARE,
                GAME_SQUARE,
                GAME_SQUARE,
            )
            pygame.draw.rect(screen, LIGHT if (row + col) % 2 == 0 else DARK, rect)
            if last_move and (row, col) in (last_move.start, last_move.end):
                highlight = pygame.Surface((GAME_SQUARE, GAME_SQUARE), pygame.SRCALPHA)
                highlight.fill((*LAST_MOVE, 145))
                screen.blit(highlight, rect.topleft)
            if selected == (row, col):
                pygame.draw.rect(screen, (255, 244, 143), rect, width=5)
            if any(move.end == (row, col) for move in legal):
                pygame.draw.circle(screen, (53, 87, 62), rect.center, 12)
            piece = board.grid[row][col]
            if piece:
                # A small shadow and high-contrast ink make the system chess glyphs
                # substantially clearer than the previous flat rendering.
                glyph = PIECES[piece]
                shadow = piece_font.render(glyph, True, (30, 30, 36))
                ink = (249, 247, 238) if piece[0] == WHITE else (37, 42, 49)
                image = piece_font.render(glyph, True, ink)
                center = rect.center
                screen.blit(shadow, shadow.get_rect(center=(center[0] + 2, center[1] + 3)))
                screen.blit(image, image.get_rect(center=center))

    pygame.draw.rect(screen, GAME_BORDER, GAME_BOARD_RECT, width=3)
    for index in range(8):
        file_label = "abcdefgh"[index if view_color == WHITE else 7 - index]
        rank_label = str(8 - index if view_color == WHITE else index + 1)
        screen.blit(small.render(file_label, True, GAME_MUTED), (GAME_BOARD_LEFT + index * GAME_SQUARE + 5, GAME_BOARD_TOP + GAME_BOARD_SIZE - 21))
        screen.blit(small.render(rank_label, True, GAME_MUTED), (GAME_BOARD_LEFT + 5, GAME_BOARD_TOP + index * GAME_SQUARE + 3))


def promotion_rects():
    left, top, size, gap = 253, 363, 105, 13
    return {kind: pygame.Rect(left + index * (size + gap), top, size, size) for index, kind in enumerate("QRBN")}


def draw_promotion_picker(screen, piece_font, title_font, small, color):
    shade = pygame.Surface((GAME_WIDTH, GAME_HEIGHT), pygame.SRCALPHA)
    shade.fill((8, 11, 16, 185))
    screen.blit(shade, (0, 0))
    panel = pygame.Rect(211, 283, 578, 253)
    pygame.draw.rect(screen, GAME_PANEL, panel, border_radius=20)
    pygame.draw.rect(screen, (230, 202, 116), panel, width=3, border_radius=20)
    title = title_font.render("Choose a promotion", True, GAME_TEXT)
    screen.blit(title, title.get_rect(center=(GAME_WIDTH // 2, 326)))
    for kind, rect in promotion_rects().items():
        pygame.draw.rect(screen, (232, 226, 208), rect, border_radius=12)
        pygame.draw.rect(screen, GAME_BORDER, rect, width=2, border_radius=12)
        image = piece_font.render(PIECES[color + kind], True, (38, 43, 50) if color == BLACK else (103, 108, 116))
        screen.blit(image, image.get_rect(center=(rect.centerx, rect.centery - 8)))
        label = {"Q": "QUEEN", "R": "ROOK", "B": "BISHOP", "N": "KNIGHT"}[kind]
        caption = small.render(label, True, (44, 50, 61))
        screen.blit(caption, caption.get_rect(center=(rect.centerx, rect.bottom + 20)))
    return promotion_rects()


def draw_game_screen(screen, board, selected, legal, fonts, view_color, last_move, status, names, shown_moves, total_moves, resigned, promotion_choices):
    title_font, small, button_font, piece_font = fonts
    screen.fill(GAME_BG)
    # Subtle board-like texture behind the panels.
    for y in range(0, GAME_HEIGHT, 40):
        for x in range(0, GAME_WIDTH, 40):
            if (x // 40 + y // 40) % 2 == 0:
                pygame.draw.rect(screen, (43, 35, 55), (x, y, 40, 40))

    top_name, top_color = names["top"]
    bottom_name, bottom_color = names["bottom"]
    player_panel(
        screen, pygame.Rect(46, 42, 600, 76), top_name, top_color,
        material_difference(board, top_color), button_font, small,
    )
    exit_rect = pygame.Rect(795, 47, 155, 54)
    mouse = pygame.mouse.get_pos()
    game_button(screen, exit_rect, "EXIT", button_font, mouse)
    draw_game_board(screen, board, selected, legal, piece_font, small, view_color, last_move)

    side = pygame.Rect(682, 142, 268, 600)
    pygame.draw.rect(screen, GAME_PANEL_DARK, side, border_radius=18)
    pygame.draw.rect(screen, GAME_BORDER, side, width=2, border_radius=18)
    heading = button_font.render("MATCH", True, GAME_TEXT)
    screen.blit(heading, heading.get_rect(center=(side.centerx, side.y + 38)))
    status_lines = [status] if len(status) <= 28 else [status[:28], status[28:]]
    for index, line in enumerate(status_lines):
        status_image = small.render(line, True, GAME_MUTED)
        screen.blit(status_image, status_image.get_rect(center=(side.centerx, side.y + 78 + index * 20)))
    pygame.draw.line(screen, GAME_BORDER, (side.x + 20, side.y + 122), (side.right - 20, side.y + 122), width=1)

    previous_rect = pygame.Rect(708, 300, 216, 54)
    next_rect = pygame.Rect(708, 369, 216, 54)
    resign_rect = pygame.Rect(708, 640, 216, 58)
    game_button(screen, previous_rect, "‹  PREVIOUS MOVE", small, mouse, disabled=shown_moves == 0)
    game_button(screen, next_rect, "NEXT MOVE  ›", small, mouse, disabled=shown_moves == total_moves)
    timeline = small.render(f"Move {shown_moves} / {total_moves}", True, GAME_TEXT)
    screen.blit(timeline, timeline.get_rect(center=(side.centerx, 462)))
    hint = small.render("Review mode is read-only", True, GAME_MUTED)
    screen.blit(hint, hint.get_rect(center=(side.centerx, 492)))
    game_button(
        screen, resign_rect, "RESIGN", button_font, mouse,
        disabled=resigned is not None or board.outcome() is not None,
        danger=True,
    )

    player_panel(
        screen, pygame.Rect(46, 766, 600, 76), bottom_name, bottom_color,
        material_difference(board, bottom_color), button_font, small,
    )
    promotion_options = None
    if promotion_choices:
        promotion_options = draw_promotion_picker(screen, piece_font, button_font, small, promotion_choices[0].start and board.piece_at(promotion_choices[0].start)[0])
    return exit_rect, previous_rect, next_rect, resign_rect, promotion_options


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
    screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
    game_fonts = (
        pygame.font.SysFont("georgia", 26, bold=True),
        pygame.font.SysFont("arial", 17, bold=True),
        pygame.font.SysFont("georgia", 21, bold=True),
        pygame.font.SysFont("segoeuisymbol", 63),
    )
    board, selected, legal = Board(), None, []
    model = QTableStore() if opponent_mode == "ai" else None
    computer_color = opponent(chosen_color) if opponent_mode == "ai" else None
    next_computer_move_at = pygame.time.get_ticks() + 450
    random_source = __import__("random").Random()
    shown_moves = 0
    promotion_choices = None
    pending_learning = None
    resigned = None
    # The person using this screen is always shown underneath the board; this
    # remains true when they choose to play Black and the board is rotated.
    names = (
        {"top": ("Player 2", opponent(chosen_color)), "bottom": ("Player 1", chosen_color)}
        if opponent_mode == "player"
        else {"top": ("Computer", computer_color), "bottom": ("You", chosen_color)}
    )

    def leave(result):
        if model:
            model.close()
        return result

    def captured_piece(move):
        if move.en_passant:
            return board.piece_at((move.start[0], move.end[1]))
        return board.piece_at(move.end)

    def learn_from_player_reply(opponent_capture=None, result=None):
        nonlocal pending_learning
        if not pending_learning:
            return
        state, computer_move, computer_capture = pending_learning
        record_game_transition(
            model, computer_color, state, computer_move, computer_capture, board,
            captured_by_opponent=opponent_capture, result=result,
        )
        pending_learning = None

    while True:
        total_moves = len(board.history)
        shown_moves = min(shown_moves, total_moves)
        display_board = board if shown_moves == total_moves else replay_board(board.history, shown_moves)
        last_move = board.history[shown_moves - 1] if shown_moves else None
        if resigned:
            winner = "Black" if resigned == WHITE else "White"
            status = ("White" if resigned == WHITE else "Black") + " resigned — " + winner + " wins"
        elif shown_moves != total_moves:
            status = "Viewing an earlier position"
        elif board.outcome():
            status = board.outcome()
        elif board.in_check(board.turn):
            status = ("White" if board.turn == WHITE else "Black") + " is in check"
        else:
            status = "White to move" if board.turn == WHITE else "Black to move"

        exit_rect, previous_rect, next_rect, resign_rect, promotion_options = draw_game_screen(
            screen, display_board, selected if shown_moves == total_moves else None,
            legal if shown_moves == total_moves else [], game_fonts, chosen_color, last_move,
            status, names, shown_moves, total_moves, resigned, promotion_choices,
        )
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return leave("quit")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return leave("menu")
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                board.reset(); selected, legal, shown_moves, promotion_choices, pending_learning, resigned = None, [], 0, None, None, None
            if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
                continue
            if exit_rect.collidepoint(event.pos):
                return leave("menu")
            if previous_rect.collidepoint(event.pos) and shown_moves:
                shown_moves -= 1
                selected, legal = None, []
                continue
            if next_rect.collidepoint(event.pos) and shown_moves < total_moves:
                shown_moves += 1
                selected, legal = None, []
                continue
            if resign_rect.collidepoint(event.pos) and not resigned and not board.outcome():
                resigned = board.turn
                winner = opponent(resigned)
                winner_name = "White" if winner == WHITE else "Black"
                learn_from_player_reply(result=f"checkmate: {winner_name} wins")
                selected, legal, promotion_choices = None, [], None
                continue
            if promotion_choices:
                for kind, rect in promotion_options.items():
                    if rect.collidepoint(event.pos):
                        move = next(move for move in promotion_choices if move.promotion == kind)
                        opponent_capture = captured_piece(move)
                        board.push(move)
                        learn_from_player_reply(opponent_capture)
                        shown_moves = len(board.history)
                        selected, legal, promotion_choices = None, [], None
                        break
                continue

            human_turn = opponent_mode == "player" or board.turn == chosen_color
            if (
                not GAME_BOARD_RECT.collidepoint(event.pos)
                or resigned
                or board.outcome()
                or not human_turn
            ):
                continue
            if shown_moves != total_moves:
                # A click in the board while reviewing never alters the past.
                shown_moves = total_moves
                selected, legal = None, []
                continue
            view_square = (
                (event.pos[1] - GAME_BOARD_TOP) // GAME_SQUARE,
                (event.pos[0] - GAME_BOARD_LEFT) // GAME_SQUARE,
            )
            square = view_to_board(view_square, chosen_color)
            choices = [move for move in legal if move.end == square]
            if choices:
                if choices[0].promotion:
                    promotion_choices = choices
                else:
                    opponent_capture = captured_piece(choices[0])
                    board.push(choices[0])
                    learn_from_player_reply(opponent_capture)
                    shown_moves = len(board.history)
                    selected, legal = None, []
            elif (piece := board.piece_at(square)) and piece[0] == board.turn:
                selected = square
                legal = [move for move in board.legal_moves() if move.start == square]
            else:
                selected, legal = None, []
        if (
            opponent_mode == "ai"
            and not resigned
            and not promotion_choices
            and not board.outcome()
            and board.turn == computer_color
            and pygame.time.get_ticks() >= next_computer_move_at
        ):
            computer_state = state_id(board)
            computer_move = best_move(board, computer_color, model, random_source)
            computer_capture = captured_piece(computer_move)
            board.push(computer_move)
            if board.outcome():
                record_game_transition(
                    model, computer_color, computer_state, computer_move, computer_capture, board
                )
            else:
                pending_learning = (computer_state, computer_move, computer_capture)
            # If a human was inspecting an older position, an incoming move
            # returns the board to the live position as requested.
            shown_moves = len(board.history)
            selected, legal = None, []
            next_computer_move_at = pygame.time.get_ticks() + 450
        pygame.display.flip()
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
                        screen = pygame.display.set_mode((WINDOW, WINDOW))
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
