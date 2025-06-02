from HexGrid import create_hex_grid
import matplotlib.pyplot as plt

class Effect:
    def __init__(self, name):
        self.name = name
        self.color = None
        self.symbol = None
        self.initialize()

    def initialize(self):
        match self.name:
            case 'Trap 1':
                self.color = '#CC99CC' # Light purple
                self.symbol = '\u229d' # ⊝ Unicode Circled Dash
            case 'Trap 2':
                self.color = '#CC99CC' # Light purple
                self.symbol = '\u2295' # ⊕ Unicode Circled Plus
            case 'Trap 3':
                self.color = '#CC99CC' # Light purple
                self.symbol = '\u2297' # ⊗ Unicode Circled Times
            case 'Trap 4':
                self.color = '#CC99CC' # Light purple
                self.symbol = '\u2298' # ⊘ Circled Division Slash
            case 'Reward 1':
                self.color = '#66B2B2' # Light teal
                self.symbol = "\u229E" # ⊞ Unicode Squared Plus 
            case 'Reward 2':
                self.color = '#66B2B2' # Light teal
                self.symbol = "\u22a0" # ⊠ Unicode Squared Times 
            case "Treasure":
                self.color = '#FFCC66' # Light orange 
            case 'Obstacle':
                self.color = '#808080' # Grey
            case _:
                self.color = '#FFFFFF' # White
        

class HexRoom:
    def __init__(self, room_idx, parent=None):
        self.room_idx = room_idx
        self.parent = parent
        self.neighbors = []
        self.effect = Effect('None')  # Default effect
        
    def add_neighbors(self, neighbor_room):
        self.neighbors.append(neighbor_room)

    def setEffect(self, effect):
        self.effect = Effect(effect)

class TreasureHunt:
    def __init__(self, nrow, ncol):
        self.nrow = nrow
        self.ncol = ncol
        self.rooms = {}
        self.create_rooms()
        self.current_room = (0, 0) 
        self.available_treasures = self.find_treasures()
        self.collected_treasures = []
        self.energy_multiplier = 1.0  
        self.speed_multiplier = 1.0   
        self.last_direction = None # For Trap 3
        self.total_cost: float = 0.0
        
    def find_treasures(self):
        treasures = []
        for pos, room in self.rooms.items():
            if room.effect.name == 'Treasure':
                treasures.append(pos)
        return treasures

    def create_rooms(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                room_idx = (row, col)
                self.rooms[room_idx] = HexRoom(room_idx)

    def setEffect(self, effect):
        for room_idx, effect_value in effect.items():
            if room_idx in self.rooms:
                self.rooms[room_idx].setEffect(effect_value)

    def expandNeighbor(self, room):
        if room in self.rooms:
            row, col = room.room_idx
            if row < self.nrow - 1:
                neighbor_idx = (row + 1, col)
                if neighbor_idx in self.rooms:
                    room.add_neighbors(self.rooms[neighbor_idx])

    def getVisualizationAttributes(self, rooms, current_room):
        colors = {}
        symbols = {}
        for room_idx, room in rooms.items():
            if room.effect is not None:
                colors[room_idx] = room.effect.color
                symbols[room_idx] = room.effect.symbol
            if room_idx == current_room:
                colors[room_idx] = '#008000' # Highlight current room in green
        return colors, symbols
                

    def visualize(self):
        colors, symbols= self.getVisualizationAttributes(self.rooms, self.current_room)

        def transform_row(row, total_rows):
            return total_rows - 1 - row
       

        transformed_symbols = {
            (transform_row(row, self.nrow), col): symbol
            for (row, col), symbol in symbols.items()
        }

        transformed_colors = {
            (transform_row(row, self.nrow), col): color
            for (row, col), color in colors.items()
        }

        fig, ax = create_hex_grid(self.nrow, self.ncol, hex_size=1,
                                colors=transformed_colors,
                                symbols=transformed_symbols)

        plt.show() 
    


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

treasureHunt.visualize()
