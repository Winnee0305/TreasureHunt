from src.models.game_state import GameState
from src.models.node import Node
from src.models.path_evaluation_info import PathEvaluationInfo
from typing import List, Tuple, Set
import heapq
from abc import ABC, abstractmethod

# Superclass for A* search algorithm (Allows for different implementations)
class AStarSearch(ABC):
    def __init__(self, state):
        
        self.state = state # State of the application, e.g. maze or game state

    @abstractmethod
    def _get_neighbors(self, x, y):
        """Get neighbors of a node"""
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def _get_successors(self, state):
        """Get all possible successor states with their costs"""
        raise NotImplementedError("This method should be implemented by subclasses")

    @abstractmethod
    def _heuristic(self, state, all_states):
        """Heuristic function for A*"""
        raise NotImplementedError("This method should be implemented by subclasses")
    
    @abstractmethod
    def solve(self):
        """Solve the problem using A* algorithm"""
        raise NotImplementedError("This method should be implemented by subclasses")

# A Star algorithm for solving the treasure hunt problem in a hexagonal grid maze
class AStarTreasureHunt(AStarSearch):
    def __init__(self, maze): 
        super().__init__(maze)
        self.start_position = maze.starting_room # Starting room of the maze
        self.treasures = self._find_treasures() # Extract all treasure locations from the maze
        
    def _find_treasures(self) -> Set[Tuple[int, int]]: # Find all treasure positions in the maze
        treasures = set()
        for pos, room in self.state.rooms.items():
            if room.effect.name == 'Treasure': # Check if the room has a treasure
                treasures.add(pos)
        return treasures

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]: # Get neighbors of a cell in odd-q offset coordinates
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
        
        for dr, dc in directions: # Calculate new row and column based on direction
            new_row, new_col = row + dr, col + dc # new position based on direction
            if (0 <= new_row < self.state.nrow and 
                0 <= new_col < self.state.ncol and
                (new_row, new_col) in self.state.rooms): # Check if the new position is within bounds and exists in the maze
                # Check if it's not an obstacle
                room = self.state.rooms[(new_row, new_col)]
                if room.effect.name != 'Obstacle':
                    neighbors.append((new_row, new_col)) # Add valid neighbor to the list
        return neighbors
    
    def _calculate_movement_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Tuple[int, int]:
        return (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]) # Calculate the direction vector from one position to another
    
    def _apply_trap3_effect(self, current_pos: Tuple[int, int], direction: Tuple[int, int]) -> Tuple[int, int]: # Apply the Trap 3 effect which teleports the player
        if direction is None: # If no direction is provided, return the current position
            return current_pos
        
        intermediate_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1]) # Move one step in the given direction
        
        # Check if intermediate position is valid and not an obstacle
        if not (0 <= intermediate_pos[0] < self.state.nrow and 
                0 <= intermediate_pos[1] < self.state.ncol and
                intermediate_pos in self.state.rooms and
                self.state.rooms[intermediate_pos].effect.name != 'Obstacle'):
            return current_pos  # Can't move at all
        
        # Try to move second step
        final_pos = (intermediate_pos[0] + direction[0], intermediate_pos[1] + direction[1])
        
        # Check if final position is valid and not an obstacle
        if (0 <= final_pos[0] < self.state.nrow and 
            0 <= final_pos[1] < self.state.ncol and
            final_pos in self.state.rooms and
            self.state.rooms[final_pos].effect.name != 'Obstacle'):
            return final_pos
        else:
            return intermediate_pos  # Can only move one step
        
    def _get_successors(self, state: GameState) -> List[Tuple[GameState, float]]:
        """Get all possible successor states with their costs"""
        successors = []
        
        # Check if we're in a forced movement state
        if state.forced_steps_remaining > 0:
            # Can only move in the forced direction
            return self._get_forced_movement_successors(state)
        
        # Normal movement - get all neighbors
        neighbors = self._get_neighbors(state.position[0], state.position[1])
        step = 0
        for next_pos in neighbors:
            # Calculate movement cost based on current state's multipliers
            movement_cost = 1.0 * state.energy_multiplier
            
            # Add significant cost for stepping on unactivated traps
            if next_pos not in state.activated_effects and next_pos in self.state.rooms:
                room = self.state.rooms[next_pos]
                effect = room.effect.name
                if effect == 'Trap 1':
                    movement_cost *= 4.0  # Severe penalty for energy doubling
                elif effect == 'Trap 2':
                    movement_cost *= 3.0  # Severe penalty for time doubling
                elif effect == 'Trap 3':
                    movement_cost *= 1  # Severe penalty for forced movement
                elif effect == 'Trap 4':
                    movement_cost *= 5.0  # Very severe penalty for treasure removal
            
            # Create new state starting with current state values
            new_state = GameState(
                position=next_pos,
                step = step + 1,
                collected_treasures=state.collected_treasures.copy(),
                available_treasures=state.available_treasures.copy(),
                activated_effects=state.activated_effects.copy(),
                energy_multiplier=state.energy_multiplier,
                last_direction=(next_pos[0] - state.position[0], next_pos[1] - state.position[1]),
                total_cost=state.total_cost + movement_cost,
                forced_steps_remaining=0,  # Reset forced movement
                forced_direction=None
            )
            
            # Apply effects of the destination cell AFTER moving there
            room = self.state.rooms[next_pos]
            effect_name = room.effect.name
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
                # Double time to move (original cost = 1 add one more step)
                step += 1
                new_state.activated_effects.add(next_pos)
                
            elif effect_name == 'Trap 3' and not effect_already_used:
                # Activate Trap 3: set up forced movement for next 2 steps
                new_state.activated_effects.add(next_pos)
                new_state.forced_steps_remaining = 2
                new_state.forced_direction = new_state.last_direction
                
            elif effect_name == 'Trap 4' and not effect_already_used:
                # Remove all uncollected treasures
                new_state.available_treasures.clear()
                new_state.activated_effects.add(next_pos)
                
            elif effect_name == 'Reward 1' and not effect_already_used:
                # Halve energy consumption
                new_state.energy_multiplier = max(0.125, new_state.energy_multiplier / 2.0)
                new_state.activated_effects.add(next_pos)
                
            elif effect_name == 'Reward 2' and not effect_already_used:
                # Halve step to move (original cost = 1 add one more step)
                step -= 0.5
                new_state.activated_effects.add(next_pos)
            
            successors.append((new_state, movement_cost))
        
        return successors
    
    def _get_forced_movement_successors(self, state: GameState) -> List[Tuple[GameState, float]]:
        """Get successors when in forced movement mode (Trap 3 effect)"""
        successors = []
        
        if state.forced_direction is None:
            # No forced direction, can't move (shouldn't happen)
            return successors
        
        # Calculate forced position
        forced_pos = (
            state.position[0] + state.forced_direction[0],
            state.position[1] + state.forced_direction[1]
        )
        
        # Check if forced position is valid and not an obstacle
        if (0 <= forced_pos[0] < self.state.nrow and 
            0 <= forced_pos[1] < self.state.ncol and
            forced_pos in self.state.rooms and
            self.state.rooms[forced_pos].effect.name != 'Obstacle'):
            
            # Calculate movement cost
            movement_cost = 1.0 * state.energy_multiplier 
            
            # Create new state after forced movement
            new_state = GameState(
                position=forced_pos,
                step = state.step,
                collected_treasures=state.collected_treasures.copy(),
                available_treasures=state.available_treasures.copy(),
                activated_effects=state.activated_effects.copy(),
                energy_multiplier=state.energy_multiplier,
                last_direction=state.forced_direction,
                total_cost=state.total_cost + movement_cost,
                forced_steps_remaining=state.forced_steps_remaining - 1,  # Decrease counter
                forced_direction=state.forced_direction  # Keep same direction
            )
            
            # Apply effects at the forced position
            room = self.state.rooms[forced_pos]
            effect_name = room.effect.name
            effect_already_used = forced_pos in state.activated_effects
            
            if effect_name == 'Treasure' and forced_pos in new_state.available_treasures:
                new_state.collected_treasures.add(forced_pos)
                new_state.available_treasures.remove(forced_pos)
                
            elif effect_name == 'Trap 1' and not effect_already_used:
                new_state.energy_multiplier *= 2.0
                new_state.activated_effects.add(forced_pos)
                
            elif effect_name == 'Trap 2' and not effect_already_used:
                new_state.activated_effects.add(forced_pos)
                
            elif effect_name == 'Trap 3' and not effect_already_used:
                # Landing on another Trap 3 during forced movement
                new_state.activated_effects.add(forced_pos)
                new_state.forced_steps_remaining = 2
                new_state.forced_direction = new_state.last_direction
                
            elif effect_name == 'Trap 4' and not effect_already_used:
                new_state.available_treasures.clear()
                new_state.activated_effects.add(forced_pos)
                
            elif effect_name == 'Reward 1' and not effect_already_used:
                new_state.energy_multiplier = max(0.125, new_state.energy_multiplier / 2.0)
                new_state.activated_effects.add(forced_pos)
                
            elif effect_name == 'Reward 2' and not effect_already_used:
                new_state.activated_effects.add(forced_pos)
            
            successors.append((new_state, movement_cost))
        
        # If forced movement is blocked, the player is stuck (no successors)
        # This represents being unable to complete the forced movement
        
        return successors

    def _heuristic(self, state: GameState) -> float:
        """
        Admissible and consistent heuristic function for A*.
        Evaluates paths to treasures considering obstacles and traps.
        """
        uncollected_treasures = state.available_treasures - state.collected_treasures
        if not uncollected_treasures:
            return 0.0
        
        # Base distance calculation
        current_pos = state.position
        min_cost = float('inf')
        
        # Evaluate path to each treasure
        for treasure_pos in uncollected_treasures:
            # Calculate hex grid distance
            dx = abs(treasure_pos[1] - current_pos[1])
            dy = abs(treasure_pos[0] - current_pos[0])
            base_distance = max(dx, dy + dx/2)
            
            # Count obstacles and traps in the approximate path
            path_cost = self._evaluate_path_to_treasure(current_pos, treasure_pos, state.activated_effects)
            
            # Total cost for this treasure
            total_cost = (base_distance + path_cost) * state.energy_multiplier
            min_cost = min(min_cost, total_cost)
        
        # Add penalty for forced movement if active
        if state.forced_steps_remaining > 0:
            min_cost += state.forced_steps_remaining * state.energy_multiplier
        
        return max(0.0, min_cost)

    def _evaluate_path_to_treasure(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int], activated_effects: Set[Tuple[int, int]]) -> float:
        """
        Evaluate the difficulty of the path between two positions.
        Returns a cost estimate that considers obstacles and traps.
        """
        # Get points in a rough rectangle between start and end
        x1, y1 = start_pos
        x2, y2 = end_pos
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        path_cost = 0.0
        obstacle_count = 0
        
        # Check each position in the approximate path area
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                pos = (x, y)
                if pos not in self.state.rooms:
                    continue
                    
                # Skip already activated effects
                if pos in activated_effects:
                    continue
                
                room = self.state.rooms[pos]
                effect = room.effect.name
                
                # Add costs based on room effects
                if effect == 'Obstacle':
                    obstacle_count += 1
                elif effect == 'Trap 1':
                    path_cost += 0.5
                elif effect == 'Trap 2':
                    path_cost += 0.5
                elif effect == 'Trap 3':
                    path_cost += 0.75
                elif effect == 'Trap 4':
                    path_cost += 1.0
                elif effect == 'Reward 1':
                    path_cost -= 0.25
                elif effect == 'Reward 2':
                    path_cost -= 0.25
        
        # If too many obstacles, increase cost significantly
        if obstacle_count > 1:
            path_cost += obstacle_count * 2.0
            
        return max(0.0, path_cost)

    def solve(self) -> Tuple[List[GameState], float, List[PathEvaluationInfo]]:
        """Solve the treasure hunt using A* algorithm"""
        initial_state = GameState(
            position=self.start_position,
            step=0,
            collected_treasures=set(),
            available_treasures=self.treasures.copy(),
            activated_effects=set(),
            total_cost=0.0,
            forced_steps_remaining=0,
            forced_direction=None
        )
        
        # Priority queue: (f_score, tie_breaker, g_score, state)
        tie_breaker = 0
        initial_h = self._heuristic(initial_state)
        initial_node = Node(initial_state, initial_h, 0.0, initial_h)
        open_set = [(initial_h, tie_breaker, 0.0, initial_node)]
        closed_set = set()
        came_from = {}
        g_score = {initial_state: 0.0}
        
        # Store evaluation information for nodes in the path
        path_evaluations = {}
        
        print("\n=== Starting A* Search ===")
        print(f"Initial position: {initial_state.position}")
        print(f"Treasures to collect: {self.treasures}")
        
        while open_set:
            current_f, _, current_g, current_node = heapq.heappop(open_set)
            current_state = current_node.state
            
            print(f"\n--- Expanding Node ---")
            print(f"Position: {current_state.position}")
            print(f"f(n)={current_f:.2f}, g(n)={current_g:.2f}, h(n)={current_f-current_g:.2f}")
            print(f"Collected treasures: {current_state.collected_treasures}")
            print(f"Activated effects: {current_state.activated_effects}")
            
            if current_state in closed_set:
                print("Already explored, skipping...")
                continue
                
            closed_set.add(current_state)
            
            # Check if we've collected all treasures
            if len(current_state.collected_treasures) == len(self.treasures):
                print("\n=== SOLUTION FOUND ===")
                print(f"Final position: {current_state.position}")
                print(f"Total cost: {current_g:.2f}")
                print("Reconstructing path...")
                
                path = []
                state = current_state
                while state in came_from:
                    prev_state = came_from[state]
                    print(f"\nAdding to solution path:")
                    print(f"  From: {prev_state.position} -> To: {state.position}")
                    print(f"  Effect at destination: {self.state.rooms[state.position].effect.name}")
                    print(f"  Treasures collected: {state.collected_treasures}")
                    print(f"  Effects activated: {state.activated_effects}")
                    path.append(state)
                    state = prev_state
                    
                path.append(initial_state)
                path.reverse()
                
                print("\n=== Final Path Summary ===")
                print("Position sequence:")
                for i, state in enumerate(path):
                    effect = self.state.rooms[state.position].effect.name
                    print(f"{i}. {state.position} ({effect})")
                
                return path, current_g, []
            nodes_to_add = []
            # Explore successors
            print("\nExploring successors:")
            for next_state, cost in self._get_successors(current_state):
                if next_state in closed_set:
                    continue
                
                tentative_g = current_g + cost
                h_score = self._heuristic(next_state)
                f_score = tentative_g + h_score
                
                print(f"\nSuccessor at {next_state.position}:")
                print(f"  Effect: {self.state.rooms[next_state.position].effect.name}")
                print(f"  f(n)={f_score:.2f}, g(n)={tentative_g:.2f}, h(n)={h_score:.2f}")
                
                if next_state not in g_score or tentative_g < g_score[next_state]:
                    came_from[next_state] = current_state
                    g_score[next_state] = tentative_g
                    tie_breaker += 1
                    next_node = Node(next_state, f_score, tentative_g, h_score)
                    nodes_to_add.append((f_score, tie_breaker, tentative_g, next_node))
                    print(f"  → Added to open set (better path found)")
                    if next_state.position in self.treasures:
                        print(f"  → Contains treasure!")
                else:
                    print(f"  → Skipped (existing path is better)")
            
            # Add new nodes to open set
            for node in nodes_to_add:
                heapq.heappush(open_set, node)
            
            # Show current open set
            print("\nOpen set after expansion (top 5 by f-score):")
            sorted_open = sorted(open_set, key=lambda x: (x[0], x[2]))
            for f, _, g, node in sorted_open[:5]:
                pos = node.state.position
                effect = self.state.rooms[pos].effect.name
                print(f"  {pos} ({effect}): f={f:.2f}, g={g:.2f}, h={f-g:.2f}")
            if len(sorted_open) > 5:
                print(f"  ... and {len(sorted_open)-5} more nodes")
            print("-" * 80)
        
        print("\nNo solution found!")
        return [], float('inf'), []

    def visualize_solution(self, path: List[GameState], evaluation_history: List[PathEvaluationInfo] = None):
        """Visualize the solution path with enhanced validation and node evaluation information"""
        if not path:
            print("No solution found!")
            return
        
        print(f"Solution found with {len(path)-1} steps!") # -1 because the initial state is not included in the path
        print(f"Total cost: {path[-1].total_cost:.2f}")
        print(f"Treasures collected: {len(path[-1].collected_treasures)}")

        print("\nDetailed Path:")
        for i, state in enumerate(path):
            pos = state.position
            effect = self.state.rooms[pos].effect.name
            treasures = len(state.collected_treasures)
            
            # Calculate energy cost for this step
            if i > 0:
                prev_state = path[i-1]
                step_cost = state.total_cost - prev_state.total_cost
                energy_mult = prev_state.energy_multiplier
                print(f"Step {i}: {pos} (Effect: {effect}, Treasures: {treasures}, "
                    f"Step Cost: {step_cost:.2f}, Energy Mult: {energy_mult:.2f}," 
                    f"Total Cost: {state.total_cost:.2f}")
            else:
                print(f"Step {i}: {pos} (Effect: {effect}, Treasures: {treasures}, "
                    f"Energy Mult: {state.energy_multiplier:.2f}")
        
        # Visualize the path
        for i in range(len(path)):
            self.state.visualize(path[i])


