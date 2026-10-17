import os
from typing import Optional

import chess
import chess.engine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_PATH = os.getenv("STOCKFISH_PATH", "stockfish")

app = FastAPI(title="Chess AI Assistant API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    fen: str = Field(default=chess.STARTING_FEN, min_length=10)
    think_time: float = Field(default=0.5, ge=0.1, le=3.0)

class AnalyzeResponse(BaseModel):
    fen: str
    best_move: str
    san: str
    from_square: str
    to_square: str
    evaluation_cp: Optional[int]
    evaluation: str
    explanation: str
    legal_moves: int

@app.get("/")
def root():
    return {"app": "Chess AI Assistant", "status": "ready", "endpoint": "/analyze"}

@app.get("/health")
def health():
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        engine.quit()
        return {"ok": True, "engine": "Stockfish ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Stockfish not found or failed to start. Set STOCKFISH_PATH. Details: {exc}")

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        board = chess.Board(req.fen)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid FEN string.")
    if not board.is_valid():
        raise HTTPException(status_code=400, detail="This FEN describes an invalid chess position.")
    if board.is_game_over():
        raise HTTPException(status_code=400, detail=f"Game is over: {board.result(claim_draw=True)}")

    legal_count = board.legal_moves.count()
    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
        try:
            info = engine.analyse(board, chess.engine.Limit(time=req.think_time))
            move = info.get("pv", [None])[0]
            score = info.get("score")
        finally:
            engine.quit()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not run Stockfish. Check STOCKFISH_PATH. Details: {exc}")

    if move is None:
        raise HTTPException(status_code=500, detail="Stockfish did not return a move.")
    san = board.san(move)
    cp = None
    eval_text = "Unavailable"
    if score is not None:
        pov_score = score.pov(board.turn)
        cp = pov_score.score(mate_score=100000)
        if pov_score.is_mate():
            mate = pov_score.mate()
            eval_text = f"Mate in {mate}" if mate is not None and mate > 0 else f"Mated in {abs(mate or 0)}"
        else:
            eval_text = f"{cp / 100:+.2f} from side to move"

    piece = board.piece_at(move.from_square)
    piece_name = {
        chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
        chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"
    }.get(piece.piece_type if piece else 0, "piece")
    target_piece = board.piece_at(move.to_square)
    if target_piece:
        explanation = f"Stockfish recommends {san}: your {piece_name} captures the piece on {chess.square_name(move.to_square)}. Check the resulting position before playing."
    elif board.gives_check(move):
        explanation = f"Stockfish recommends {san}: this {piece_name} move gives check and forces the opponent to respond."
    else:
        explanation = f"Stockfish recommends {san}: it is the engine's strongest move in this position at the selected search time. The arrow shows {chess.square_name(move.from_square)} → {chess.square_name(move.to_square)}."

    return AnalyzeResponse(
        fen=board.fen(), best_move=move.uci(), san=san,
        from_square=chess.square_name(move.from_square),
        to_square=chess.square_name(move.to_square),
        evaluation_cp=cp, evaluation=eval_text, explanation=explanation,
        legal_moves=legal_count,
    )