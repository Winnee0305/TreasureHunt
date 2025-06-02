from TreasureHunt import TreasureHunt
from Algorithm import AStarTreasureHunt

if __name__ == "__main__":
    treasureHunt = TreasureHunt(6, 10)

    treasureHunt.setEffect({
        (3, 0): 'Obstacle',
        (1, 1): 'Trap 2',
        (3, 1): 'Reward 1',
        (2, 2): 'Obstacle',
        (1, 3): 'Trap 4',
        (3, 3): 'Obstacle',
        (4, 2): 'Trap 2',
        (4, 3): 'Treasure',
        (0, 4): 'Reward 1',
        (1, 4): 'Treasure',
        (2, 4): 'Obstacle',
        (4, 4): 'Obstacle',
        (3, 5): 'Trap 3',
        (5, 5): 'Reward 2',
        (1, 6): 'Trap 3',
        (3, 6): 'Obstacle',
        (4, 6): 'Obstacle',
        (2, 7): 'Reward 2',
        (3, 7): 'Treasure',
        (4, 7): 'Obstacle',
        (1, 8): 'Obstacle',
        (2, 8): 'Trap 1',
        (3, 9): 'Treasure'

    })
    solver = AStarTreasureHunt(treasureHunt)

    solution_path, total_cost, evaluation_history = solver.solve()
    
    solver.visualize_solution(solution_path, evaluation_history)

