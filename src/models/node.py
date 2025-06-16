from src.models.game_state import GameState

class Node:
    """Represents a node in the A* search"""
    def __init__(self, state: GameState, f_score: float, g_score: float, h_score: float):
        self.state = state
        self.f_score = f_score
        self.g_score = g_score
        self.h_score = h_score
        self.position = state.position

    def __lt__(self, other):
        return self.f_score < other.f_score