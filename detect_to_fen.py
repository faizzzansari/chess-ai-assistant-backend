import chess
from ultralytics import YOLO

model = YOLO("models/chess_pieces.pt")
result = model.predict(
    source="test_board.png",
    conf=0.25,
    imgsz=640,
    save=False,
    verbose=False,
)[0]

# Find the board rectangle.
board_box = None
for box in result.boxes:
    if model.names[int(box.cls[0].item())] == "board":
        board_box = [float(v) for v in box.xyxy[0].tolist()]
        break

if board_box is None:
    raise RuntimeError("Board not detected.")

x1, y1, x2, y2 = board_box
square_size = min((x2 - x1) / 8, (y2 - y1) / 8)

# Map model labels to standard FEN characters.
piece_symbols = {
    "white_king": "K", "white_queen": "Q",
    "white_rook": "R", "white_bishop": "B",
    "white_knight": "N", "white_pawn": "P",
    "black_king": "k", "black_queen": "q",
    "black_rook": "r", "black_bishop": "b",
    "black_knight": "n", "black_pawn": "p",
}

squares = {}
duplicates = []

for box in result.boxes:
    name = model.names[int(box.cls[0].item())]
    if name == "board":
        continue

    confidence = float(box.conf[0].item())
    bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0].tolist()]
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2

    file_idx = round((cx - (x1 + square_size / 2)) / square_size)
    rank_idx = round((cy - (y1 + square_size / 2)) / square_size)

    if not (0 <= file_idx < 8 and 0 <= rank_idx < 8):
        print("Outside board:", name, round(cx, 1), round(cy, 1))
        continue

    square = chess.square(file_idx, 7 - rank_idx)
    square_name = chess.square_name(square)

    if square in squares:
        duplicates.append(square_name)
        print("DUPLICATE:", square_name, squares[square][0], name)
        # Keep the higher-confidence detection.
        if confidence <= squares[square][1]:
            continue

    squares[square] = (name, confidence)

board = chess.Board(None)
for square, (name, confidence) in squares.items():
    board.set_piece_at(square, chess.Piece.from_symbol(piece_symbols[name]))

print("\nDetected pieces:", len(squares))
print("Duplicate squares:", duplicates or "None")
print("\nBoard placement:")
print(board)

# This screenshot appears to have Black to move.
# Turn, castling rights, and move counters must be confirmed separately.
board.turn = chess.BLACK
board.set_castling_fen("KQkq")
board.ep_square = None
board.halfmove_clock = 0
board.fullmove_number = 3

print("\nGenerated FEN:", board.fen())
print("Position valid:", board.is_valid())
print("Status:", board.status())
