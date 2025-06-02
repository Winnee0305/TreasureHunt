# Treasure Hunt A\* Pathfinding

This project implements an A\* pathfinding algorithm to solve a treasure hunt game on a hexagonal grid. The game includes various elements such as treasures, traps, and rewards that affect the pathfinding strategy.

## Project Structure

```
src/
├── game/            # Game logic and mechanics
│   ├── treasure_hunt.py
│   └── astar_solver.py
├── models/           # Data models and structures
│   ├── game_state.py
|   ├── hex.py
│   ├── node.py
│   └── path_evaluation_info.py
└── main.py         # Entry point

```

## Features

- Hexagonal grid implementation with odd-q offset coordinate system
- A\* pathfinding algorithm with custom heuristics
- Various game elements:
  - Treasures to collect
  - Traps with different effects
  - Rewards that modify movement costs
- Detailed visualization of:
  - Path progression
  - Node evaluation
  - Open set queue states
  - Final solution path

## Requirements

- Python 3.7+
- matplotlib

## Running the Project

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install matplotlib
   ```
3. Run the main script:
   ```bash
   python src/main.py
   ```

## Game Elements

- **Treasures**: Must be collected to complete the game
- **Traps**:
  - Trap 1: Doubles energy consumption
  - Trap 2: Doubles movement time
  - Trap 3: Forces movement in the last direction
  - Trap 4: Removes all uncollected treasures
- **Rewards**:
  - Reward 1: Halves energy consumption
  - Reward 2: Halves movement time
- **Obstacles**: Blocked cells that cannot be traversed
