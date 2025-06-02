import heapq
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
from HexGrid import create_hex_grid
import matplotlib.pyplot as plt

@dataclass
class GameState:
    """Represents the current state of the game"""
    position: Tuple[int, int]
    collected_treasures: Set[Tuple[int, int]]
    available_treasures: Set[Tuple[int, int]]
    activated_effects: Set[Tuple[int, int]]  # Track which traps/rewards have been used
    energy_multiplier: float = 1.0  # For Trap 1 and Reward 1
    speed_multiplier: float = 1.0   # For Trap 2 and Reward 2
    last_direction: Optional[Tuple[int, int]] = None
    total_cost: float = 0.0
    
    def __hash__(self):
        return hash((
            self.position,
            tuple(sorted(self.collected_treasures)),
            tuple(sorted(self.available_treasures)),
            tuple(sorted(self.activated_effects)),
            self.energy_multiplier,
            self.speed_multiplier,
            self.last_direction
        ))
    
    def __eq__(self, other):
        if not isinstance(other, GameState):
            return False
        return (
            self.position == other.position and
            self.collected_treasures == other.collected_treasures and
            self.available_treasures == other.available_treasures and
            self.activated_effects == other.activated_effects and
            abs(self.energy_multiplier - other.energy_multiplier) < 1e-6 and
            abs(self.speed_multiplier - other.speed_multiplier) < 1e-6 and
            self.last_direction == other.last_direction
        )


class AStarTreasureHunt:
    def __init__(self, maze):
        self.maze = maze
        self.start_position = (0, 0)
        self.treasures = self._find_treasures()
        
    def _find_treasures(self) -> Set[Tuple[int, int]]:
        """Find all treasure locations in the maze"""
        treasures = set()
        for pos, room in self.maze.rooms.items():
            if room.effect.name == 'Treasure':
                treasures.add(pos)
        return treasures
    
    def _get_hex_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid hexagonal grid neighbors for odd-q offset coordinate system"""
        neighbors = []
        
        # In odd-q offset coordinates, the neighbor directions depend on whether the column is odd or even
        if col % 2 == 0:  # Even column (not offset vertically)
            directions = [
                (-1, 0),   # North
                (0, -1),   # Northwest
                (0, 1),    # Northeast
                (1, 0),    # South
                (1, -1),   # Southwest
                (1, 1),    # Southeast
            ]
        else:  # Odd column (offset upward)
            directions = [
                (-1, 0),   # North
                (-1, -1),  # Northwest
                (-1, 1),   # Northeast
                (1, 0),    # South
                (0, -1),   # Southwest
                (0, 1),    # Southeast
            ]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.maze.nrow and 
                0 <= new_col < self.maze.ncol and
                (new_row, new_col) in self.maze.rooms):
                # Check if it's not an obstacle
                room = self.maze.rooms[(new_row, new_col)]
                if room.effect.name != 'Obstacle':
                    neighbors.append((new_row, new_col))
        return neighbors
    
    def _calculate_movement_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate the direction vector for movement"""
        return (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
    
    def _apply_trap3_effect(self, current_pos: Tuple[int, int], direction: Tuple[int, int]) -> Tuple[int, int]:
        """Apply Trap 3 effect - move two cells in the last direction with proper validation"""
        if direction is None:
            return current_pos
        
        # Move one step in the given direction
        intermediate_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
        
        # Check if intermediate position is valid and not an obstacle
        if not (0 <= intermediate_pos[0] < self.maze.nrow and 
                0 <= intermediate_pos[1] < self.maze.ncol and
                intermediate_pos in self.maze.rooms and
                self.maze.rooms[intermediate_pos].effect.name != 'Obstacle'):
            return current_pos  # Can't move at all
        
        # Try to move second step
        final_pos = (intermediate_pos[0] + direction[0], intermediate_pos[1] + direction[1])
        
        # Check if final position is valid and not an obstacle
        if (0 <= final_pos[0] < self.maze.nrow and 
            0 <= final_pos[1] < self.maze.ncol and
            final_pos in self.maze.rooms and
            self.maze.rooms[final_pos].effect.name != 'Obstacle'):
            return final_pos
        else:
            return intermediate_pos  # Can only move one step
        
    def _get_successors(self, state: GameState) -> List[Tuple[GameState, float]]:
        """Get all possible successor states with their costs"""
        successors = []
        neighbors = self._get_hex_neighbors(state.position[0], state.position[1])
        
        for next_pos in neighbors:
            # Calculate movement cost based on current state's multipliers
            movement_cost = 1.0 * state.energy_multiplier * state.speed_multiplier
            
            # Create new state starting with current state values
            new_state = GameState(
                position=next_pos,
                collected_treasures=state.collected_treasures.copy(),
                available_treasures=state.available_treasures.copy(),
                activated_effects=state.activated_effects.copy(),
                energy_multiplier=state.energy_multiplier,
                speed_multiplier=state.speed_multiplier,
                last_direction=self._calculate_movement_direction(state.position, next_pos),
                total_cost=state.total_cost + movement_cost
            )
            
            # Apply effects of the destination cell AFTER moving there
            room = self.maze.rooms[next_pos]
            effect_name = room.effect.name
            
            # Check if this effect has already been activated (only for traps and rewards)
            effect_already_used = next_pos in state.activated_effects
            
            if effect_name == 'Treasure' and next_pos in new_state.available_treasures:
                # Collect treasure (treasures can always be collected)
                new_state.collected_treasures.add(next_pos)
                new_state.available_treasures.remove(next_pos)
                
            elif effect_name == 'Trap 1' and not effect_already_used:
                # Double energy consumption for future moves
                new_state.energy_multiplier *= 2.0
                new_state.activated_effects.add(next_pos)
                
            elif effect_name == 'Trap 2' and not effect_already_used:
                # Double time to move (double speed multiplier)
                new_state.speed_multiplier *= 2.0
                new_state.activated_effects.add(next_pos)
                
            elif effect_name == 'Trap 3' and not effect_already_used:
                # For Trap 3, first add the state where we step onto the trap
                new_state.activated_effects.add(next_pos)
                
                # Calculate the teleported position
                current_direction = self._calculate_movement_direction(state.position, next_pos)
                trap3_pos = self._apply_trap3_effect(next_pos, current_direction)
                
                if trap3_pos != next_pos:
                    teleported_state = GameState(
                        position=trap3_pos,  # This is the actual final position
                        collected_treasures=new_state.collected_treasures.copy(),
                        available_treasures=new_state.available_treasures.copy(),
                        activated_effects=new_state.activated_effects.copy(),
                        energy_multiplier=new_state.energy_multiplier,
                        speed_multiplier=new_state.speed_multiplier,
                        last_direction=current_direction,
                        total_cost=new_state.total_cost  # No additional cost for teleportation
                    )
                    # Only add the teleported state, not the intermediate state
                    successors.append((teleported_state, movement_cost))
                    # Skip adding the original state since we're using the teleported one
                    continue
                
            elif effect_name == 'Trap 4' and not effect_already_used:
                # Remove all uncollected treasures
                new_state.available_treasures.clear()
                new_state.activated_effects.add(next_pos)
                
            elif effect_name == 'Reward 1' and not effect_already_used:
                # Halve energy consumption
                new_state.energy_multiplier = max(0.125, new_state.energy_multiplier / 2.0)
                new_state.activated_effects.add(next_pos)
                
            elif effect_name == 'Reward 2' and not effect_already_used:
                # Halve time to move
                new_state.speed_multiplier = max(0.125, new_state.speed_multiplier / 2.0)
                new_state.activated_effects.add(next_pos)
            
            # Add the state (except for Trap 3 which is handled above)
            successors.append((new_state, movement_cost))
        
        return successors

    def _heuristic(self, state: GameState) -> float:
        """Heuristic function for A* - Manhattan distance to nearest uncollected treasure"""
        uncollected_treasures = state.available_treasures - state.collected_treasures
        if not uncollected_treasures:
            return 0.0
        
        # Find minimum distance to any uncollected treasure
        min_distance = float('inf')
        current_row, current_col = state.position
        
        for treasure_row, treasure_col in uncollected_treasures:
            # Use hexagonal distance approximation
            dx = abs(treasure_col - current_col)
            dy = abs(treasure_row - current_row)
            distance = max(dx, dy + dx/2)
            min_distance = min(min_distance, distance)
        
        heuristic_value = min_distance * state.energy_multiplier * state.speed_multiplier
        return heuristic_value
    
    def solve(self) -> Tuple[List[GameState], float]:
        """Solve the treasure hunt using A* algorithm"""
        initial_state = GameState(
            position=self.start_position,
            collected_treasures=set(),
            available_treasures=self.treasures.copy(),
            activated_effects=set(),
            total_cost=0.0
        )
        
        # Priority queue: (f_score, tie_breaker, g_score, state)
        tie_breaker = 0
        open_set = [(self._heuristic(initial_state), tie_breaker, 0.0, initial_state)]
        closed_set = set()
        came_from = {}
        g_score = {initial_state: 0.0}
        
        while open_set:
            current_f, _, current_g, current_state = heapq.heappop(open_set)
            
            if current_state in closed_set:
                continue
                
            closed_set.add(current_state)
            
            # Check if we've collected all treasures (must have collected them, not just have none available)
            if len(current_state.collected_treasures) == len(self.treasures):
                # Reconstruct path
                path = []
                state = current_state
                while state in came_from:
                    path.append(state)
                    state = came_from[state]
                path.append(initial_state)
                path.reverse()
                print(f"Solution found with {len(path)} steps and total cost: {current_g:.2f}")
                return path, current_g
            
            # Explore successors
            for next_state, cost in self._get_successors(current_state):
                if next_state in closed_set:
                    continue
                
                tentative_g = current_g + cost
                
                if next_state not in g_score or tentative_g < g_score[next_state]:
                    came_from[next_state] = current_state
                    g_score[next_state] = tentative_g
                    f_score = tentative_g + self._heuristic(next_state)
                    tie_breaker += 1
                    heapq.heappush(open_set, (f_score, tie_breaker, tentative_g, next_state))
        
        # No solution found
        return [], float('inf')
    
    def visualize_solution(self, path: List[GameState]):
        """Visualize the solution path with enhanced validation"""
        if not path:
            print("No solution found!")
            return
        
        print(f"Solution found with {len(path)} steps!")
        print(f"Total cost: {path[-1].total_cost:.2f}")
        print(f"Treasures collected: {len(path[-1].collected_treasures)}")
        
        # Show path positions with detailed information
        print("\nDetailed Path:")
        for i, state in enumerate(path):
            pos = state.position
            effect = self.maze.rooms[pos].effect.name
            treasures = len(state.collected_treasures)
            
            # Calculate energy cost for this step
            if i > 0:
                prev_state = path[i-1]
                step_cost = state.total_cost - prev_state.total_cost
                energy_mult = prev_state.energy_multiplier
                speed_mult = prev_state.speed_multiplier
                print(f"Step {i}: {pos} (Effect: {effect}, Treasures: {treasures}, "
                    f"Step Cost: {step_cost:.2f}, Energy Mult: {energy_mult:.2f}, "
                    f"Speed Mult: {speed_mult:.2f})")
            else:
                print(f"Step {i}: {pos} (Effect: {effect}, Treasures: {treasures}, "
                    f"Energy Mult: {state.energy_multiplier:.2f}, "
                    f"Speed Mult: {state.speed_multiplier:.2f})")
        
        # # Enhanced path validation
        # print("\nPath Validation:")
        # i = 1
        # while i < len(path):
        #     prev_state = path[i-1]
        #     current_state = path[i]
        #     prev_pos = prev_state.position
        #     current_pos = current_state.position
            
        #     # Check if current position had Trap 3 effect
        #     current_room = self.maze.rooms[current_pos]
        #     if current_room.effect.name == 'Trap 3' and current_pos in current_state.activated_effects:
        #         # Check if next step is the teleportation
        #         if i + 1 < len(path):
        #             next_state = path[i + 1]
        #             next_pos = next_state.position
                    
        #             # Validate the teleportation
        #             direction = self._calculate_movement_direction(prev_pos, current_pos)
        #             expected_pos = self._apply_trap3_effect(current_pos, direction)
                    
        #             if next_pos == expected_pos and next_pos != current_pos:
        #                 print(f"Step {i}-{i+1}: Trap 3 teleportation from {current_pos} to {next_pos}")
        #                 i += 2  # Skip the teleportation step in validation
        #                 continue
        #             elif next_pos != current_pos:
        #                 print(f"WARNING: Trap 3 teleportation mismatch at step {i+1}")
        #                 print(f"  Expected: {expected_pos}, Actual: {next_pos}")
            
        #     # Normal movement validation
        #     neighbors = self._get_hex_neighbors(prev_pos[0], prev_pos[1])
        #     if current_pos not in neighbors:
        #         print(f"WARNING: Invalid jump from {prev_pos} to {current_pos}")
        #         print(f"Valid neighbors of {prev_pos}: {neighbors}")
            
        #     i += 1
        
        # # Check for effect reactivation
        # print("\nEffect Activation Check:")
        # all_activated = set()
        # for i, state in enumerate(path):
        #     pos = state.position
        #     room = self.maze.rooms[pos]
        #     effect_name = room.effect.name
            
        #     if effect_name in ['Trap 1', 'Trap 2', 'Trap 3', 'Trap 4', 'Reward 1', 'Reward 2']:
        #         if pos in all_activated:
        #             print(f"WARNING: Effect {effect_name} at {pos} activated multiple times (step {i})")
        #         else:
        #             all_activated.add(pos)
        for i in range(len(path)):
            self.maze.visualize(path[i])

        
    def visualize_path(self, path):
        # Create visualization with path highlighted
        colors, symbols = self.maze.getVisualizationAttributes(self.maze.rooms, path.position)
        
        # Transform coordinates for visualization
        def transform_row(row, total_rows):
            return total_rows - 1 - row
        
        transformed_symbols = {
            (transform_row(row, self.maze.nrow), col): symbol
            for (row, col), symbol in symbols.items()
        }
        
        transformed_colors = {
            (transform_row(row, self.maze.nrow), col): color
            for (row, col), color in colors.items()
        }
        
        fig, ax = create_hex_grid(self.maze.nrow, self.maze.ncol, hex_size=1,
                                colors=transformed_colors,
                                symbols=transformed_symbols)
        
        plt.title(f"A* Solution - Total Cost: {path.total_cost:.2f}", fontsize=40)
        plt.show()

  

if __name__ == "__main__":
    from TreasureHunt import treasureHunt
    
    solver = AStarTreasureHunt(treasureHunt)
    
    print("Solving treasure hunt with A* algorithm...")
    solution_path, total_cost = solver.solve()
    
    solver.visualize_solution(solution_path)