from typing import List, Tuple,Optional, Dict, Any


class PathEvaluationInfo:
    """Stores evaluation information for a node in the path"""
    def __init__(self, current_position: Tuple[int, int], current_f: float, current_g: float, current_h: float):
        self.current_position = current_position
        self.current_f = current_f
        self.current_g = current_g
        self.current_h = current_h
        self.neighbors: List[Dict[str, Any]] = []
        self.queue_after: List[Dict[str, Any]] = []
        self.next_chosen: Optional[Dict[str, Any]] = None
        self.chosen_position: Optional[Tuple[int, int]] = None


