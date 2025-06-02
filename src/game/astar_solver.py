
from ..models.game_state import GameState
from ..models.node import Node
from ..models.path_evaluation_info import PathEvaluationInfo
from typing import List, Tuple, Set
import heapq


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
            
            # Add the state
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

    def solve(self) -> Tuple[List[GameState], float, List[PathEvaluationInfo]]:
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
        initial_h = self._heuristic(initial_state)
        initial_node = Node(initial_state, initial_h, 0.0, initial_h)
        open_set = [(initial_h, tie_breaker, 0.0, initial_node)]
        closed_set = set()
        came_from = {}
        g_score = {initial_state: 0.0}
        
        # Store evaluation information for nodes in the path
        path_evaluations = {}
        
        # Track open set state for each step
        step_count = 0
        
        while open_set:
            current_f, _, current_g, current_node = heapq.heappop(open_set)
            current_state = current_node.state
            step_count += 1
            
            if current_state in closed_set:
                continue
                
            closed_set.add(current_state)
            
            # Store neighbor evaluation information for this state
            current_evaluations = PathEvaluationInfo(
                current_state.position,
                current_f,
                current_g,
                current_f - current_g
            )
            
            # Check if we've collected all treasures
            if len(current_state.collected_treasures) == len(self.treasures):
                # Store the final evaluations
                path_evaluations[current_state] = current_evaluations
                
                # Reconstruct path
                path = []
                state = current_state
                while state in came_from:
                    path.append(state)
                    state = came_from[state]
                path.append(initial_state)
                path.reverse()
                print(f"Solution found with {len(path)} steps and total cost: {current_g:.2f}")
                
                # Only keep evaluations for states in the path
                final_evaluations = []
                for i in range(len(path) - 1):
                    state = path[i]
                    if state in path_evaluations:
                        eval_info = path_evaluations[state]
                        next_state = path[i + 1]
                        eval_info.chosen_position = next_state.position
                        final_evaluations.append(eval_info)
                
                return path, current_g, final_evaluations
            
            # Track nodes to be added to open set
            nodes_to_add = []
            
            # Explore successors
            for next_state, cost in self._get_successors(current_state):
                if next_state in closed_set:
                    continue
                
                tentative_g = current_g + cost
                h_score = self._heuristic(next_state)
                f_score = tentative_g + h_score
                
                # Store neighbor evaluation information
                neighbor_info = {
                    'position': next_state.position,
                    'g_score': tentative_g,
                    'h_score': h_score,
                    'f_score': f_score,
                    'effect': self.maze.rooms[next_state.position].effect.name,
                    'treasures_collected': len(next_state.collected_treasures),
                    'energy_multiplier': next_state.energy_multiplier,
                    'speed_multiplier': next_state.speed_multiplier
                }
                current_evaluations.neighbors.append(neighbor_info)
                
                if next_state not in g_score or tentative_g < g_score[next_state]:
                    came_from[next_state] = current_state
                    g_score[next_state] = tentative_g
                    tie_breaker += 1
                    next_node = Node(next_state, f_score, tentative_g, h_score)
                    nodes_to_add.append((f_score, tie_breaker, tentative_g, next_node))
            
            # Add new nodes to open set
            for node in nodes_to_add:
                heapq.heappush(open_set, node)
            
            # Peek at the next node that will be chosen
            if open_set:
                next_f, _, next_g, next_node = open_set[0]
                current_evaluations.next_chosen = {
                    'position': next_node.state.position,
                    'f_score': next_f,
                    'g_score': next_g,
                    'h_score': next_f - next_g
                }
            
            # Capture final state of open set
            current_evaluations.queue_after = [
                {
                    'position': node.state.position,
                    'f_score': f,
                    'g_score': g,
                    'h_score': f - g
                }
                for f, _, g, node in open_set
            ]
            
            # Store evaluations for current state
            path_evaluations[current_state] = current_evaluations
        
        # No solution found
        return [], float('inf'), []

    def visualize_solution(self, path: List[GameState], evaluation_history: List[PathEvaluationInfo] = None):
        """Visualize the solution path with enhanced validation and node evaluation information"""
        if not path:
            print("No solution found!")
            return
        
        print(f"Solution found with {len(path)} steps!")
        print(f"Total cost: {path[-1].total_cost:.2f}")
        print(f"Treasures collected: {len(path[-1].collected_treasures)}")
        
        # Show path positions with detailed information
        print("\nPath Progression with Open Set States:")
        for step_info in evaluation_history:
            current_pos = step_info.current_position
            chosen_pos = step_info.chosen_position
            next_chosen = step_info.next_chosen
            
            print(f"\nAt position {current_pos}")
            print(f"Current node: f(n)={step_info.current_f:.2f}, g(n)={step_info.current_g:.2f}, h(n)={step_info.current_h:.2f}")
            
            print("\nNeighbors evaluated:")
            print("Position\t\tg(n)\t\th(n)\t\tf(n)\t\tEffect\t\tChosen\tNext")
            print("-" * 100)
            for neighbor in step_info.neighbors:
                pos = neighbor['position']
                g = neighbor['g_score']
                h = neighbor['h_score']
                f = neighbor['f_score']
                effect = neighbor['effect']
                in_final_path = "→" if pos == chosen_pos else " "
                next_expanded = "*" if next_chosen and pos == next_chosen['position'] else " "
                print(f"{pos}\t\t{g:.2f}\t\t{h:.2f}\t\t{f:.2f}\t\t{effect:<12}\t{in_final_path}\t{next_expanded}")
            
            if next_chosen:
                print(f"\nNext node to be expanded by A*: {next_chosen['position']} with f(n)={next_chosen['f_score']:.2f}")
            
            print("\nOpen Set Queue (sorted by f-score):")
            print("Position\t\tf(n)\t\tg(n)\t\th(n)")
            print("-" * 60)
            sorted_queue = sorted(step_info.queue_after, key=lambda x: x['f_score'])
            for node in sorted_queue:
                print(f"{node['position']}\t\t{node['f_score']:.2f}\t\t{node['g_score']:.2f}\t\t{node['h_score']:.2f}")
            
            print("\n" + "="*100)
        
        # Visualize the path
        for i in range(len(path)):
            self.maze.visualize(path[i])


