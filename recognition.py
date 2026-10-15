import chess
import numpy as np
import cv2
from ultralytics import YOLO

MODEL = YOLO("models/chess_pieces.pt")

PIECE_SYMBOLS = {
    "white_king": "K",
    "white_queen": "Q",
    "white_rook": "R",
    "white_bishop": "B",
    "white_knight": "N",
    "white_pawn": "P",
    "black_king": "k",
    "black_queen": "q",
    "black_rook": "r",
    "black_bishop": "b",
    "black_knight": "n",
    "black_pawn": "p",
}


def image_to_fen(image_bytes: bytes, turn: str = "black") -> dict:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("The uploaded image could not be decoded.")

    result = MODEL.predict(
        source=image,
        conf=0.40,
        imgsz=640,
        verbose=False,
    )[0]

    board_boxes = [
        box for box in result.boxes
        if MODEL.names[int(box.cls[0].item())] == "board"
    ]

    if not board_boxes:
        raise ValueError("No chessboard was detected.")

    board_box = max(
        board_boxes,
        key=lambda box: float(box.conf[0].item()),
    )

    x1, y1, x2, y2 = map(float, board_box.xyxy[0].tolist())

    board_width = x2 - x1
    board_height = y2 - y1

    if board_width <= 0 or board_height <= 0:
        raise ValueError("Invalid chessboard dimensions.")

    square_width = board_width / 8
    square_height = board_height / 8

    pieces = {}
    duplicates = []

    # Collect piece detections and assign each to its nearest square center.
    for box in result.boxes:
        name = MODEL.names[int(box.cls[0].item())]

        if name == "board":
            continue

        if name not in PIECE_SYMBOLS:
            continue

        bx1, by1, bx2, by2 = map(float, box.xyxy[0].tolist())

        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2

        file_index = int((cx - x1) / square_width)
        rank_index = int((cy - y1) / square_height)

        if not (0 <= file_index < 8 and 0 <= rank_index < 8):
            continue

        square = chess.square(file_index, 7 - rank_index)
        confidence = float(box.conf[0].item())

        if square in pieces:
            duplicates.append(chess.square_name(square))

            # Keep the more confident detection.
            if confidence <= pieces[square][1]:
                continue

        pieces[square] = (name, confidence)

    if not pieces:
        raise ValueError("No chess pieces were detected.")

    if len(pieces) > 32:
        raise ValueError(
            f"Detected {len(pieces)} unique occupied squares; "
            "a chess position cannot contain more than 32 pieces."
        )

    if turn not in ("white", "black"):
        raise ValueError("Turn must be 'white' or 'black'.")

    position = chess.Board(None)

    for square, (name, confidence) in pieces.items():
        position.set_piece_at(
            square,
            chess.Piece.from_symbol(PIECE_SYMBOLS[name]),
        )

    position.turn = turn == "white"

    # A screenshot cannot establish castling rights.
    # Use no castling rights rather than incorrectly claiming all are available.
    position.set_castling_fen("-")
    position.ep_square = None
    position.halfmove_clock = 0
    position.fullmove_number = 1

    if not position.is_valid():
        status = position.status()

        white_pawns = len(position.pieces(chess.PAWN, chess.WHITE))
        black_pawns = len(position.pieces(chess.PAWN, chess.BLACK))

        print(f"[RECOGNITION ERROR] status={status}")
        print(f"[RECOGNITION ERROR] FEN={position.board_fen()}")
        print(f"[RECOGNITION ERROR] white_pawns={white_pawns}")
        print(f"[RECOGNITION ERROR] black_pawns={black_pawns}")
        print(position)

        raise ValueError(
            f"Detected an invalid chess position (status {status}). "
            "The detected pieces or their square assignments may be incorrect."
        )

    return {
        "fen": position.fen(),
        "piece_count": len(pieces),
        "duplicates": duplicates,
        "board": str(position),
        "warning": (
            "Turn was selected manually. Castling rights, en passant, "
            "and move counters cannot be reliably inferred from a single image."
        ),
    }