# Treasure Hunt with A\* Search Algorithm

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-blueviolet?style=for-the-badge)](https://matplotlib.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Dataclasses](https://img.shields.io/badge/Dataclasses-Builtin%20Module-informational?style=for-the-badge)](https://docs.python.org/3/library/dataclasses.html)
[![Typing](https://img.shields.io/badge/Typing-Builtin%20Module-informational?style=for-the-badge)](https://docs.python.org/3/library/typing.html)
[![Heapq](https://img.shields.io/badge/Heapq-Builtin%20Module-informational?style=for-the-badge)](https://docs.python.org/3/library/heapq.html)

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
   pip install matplotlib numpy
   ```
3. Run the main script in an interactive Python environment (e.g., VS Code's interactive window, Jupyter Notebook) to properly view the plotted diagrams:
   ```bash
   python src/main.py
   ```

Alternatively, you can run it directly from your terminal, but the plots might close automatically unless you explicitly handle the plot display (e.g., by adding `plt.show()` and keeping the plot window open).

** Note: The first run may take some time as the program prints detailed information about the search process step-by-step

## Game Elements

- **Treasures**: Must be collected to complete the game
- **Traps**:
  - Trap 1: Doubles energy consumption
  - Trap 2: Doubles steps movement
  - Trap 3: Forces 2 movements in the last direction
  - Trap 4: Removes all uncollected treasures
- **Rewards**:
  - Reward 1: Halves energy consumption
  - Reward 2: Halves movement time
- **Obstacles**: Blocked cells that cannot be traversed
