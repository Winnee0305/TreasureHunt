from dataclasses import dataclass
from typing import Tuple, Set, Optional

@dataclass
class GameState:
    """Represents the current state of the game"""
    position: Tuple[int, int]
    step: int 
    collected_treasures: Set[Tuple[int, int]]
    available_treasures: Set[Tuple[int, int]]
    activated_effects: Set[Tuple[int, int]]  # Track which traps/rewards have been used
    energy_multiplier: float = 1.0  # For Trap 1 and Reward 1
    last_direction: Optional[Tuple[int, int]] = None
    total_cost: float = 0.0

    #For Trap 3
    forced_steps_remaining: int = 0  # How many forced steps left
    forced_direction: Optional[Tuple[int, int]] = None  # Direction of forced movement
    
    def __hash__(self): # Generate a unique hash for the game state
        return hash((
            self.position,
            tuple(sorted(self.collected_treasures)),
            tuple(sorted(self.available_treasures)),
            tuple(sorted(self.activated_effects)),
            self.energy_multiplier,
            self.last_direction,
            self.forced_steps_remaining,
            self.forced_direction
        ))
    
    def __eq__(self, other): # Check if two game states are equal
        if not isinstance(other, GameState):
            return False
        return ( # Compare all attributes for equality
            self.position == other.position and
            self.collected_treasures == other.collected_treasures and
            self.available_treasures == other.available_treasures and
            self.activated_effects == other.activated_effects and
            abs(self.energy_multiplier - other.energy_multiplier) < 1e-6 and
            self.last_direction == other.last_direction and
            self.forced_steps_remaining == other.forced_steps_remaining and
            self.forced_direction == other.forced_direction
        )
