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

    def setEffect(self, effect):
        self.effect = Effect(effect)

class TreasureHunt:
    def __init__(self, nrow, ncol):
        self.nrow = nrow
        self.ncol = ncol
        self.rooms = {}
        self.create_rooms()
        self.starting_room = (0, 0) 
        self.path = []

    def create_rooms(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                room_idx = (row, col)
                self.rooms[room_idx] = HexRoom(room_idx)

    def setEffect(self, effect):
        for room_idx, effect_value in effect.items():
            if room_idx in self.rooms:
                self.rooms[room_idx].setEffect(effect_value)

    def getVisualizationAttributes(self, rooms, path):
        colors = {}
        symbols = {}
        for room_idx, room in rooms.items():
            if room.effect is not None:
                colors[room_idx] = room.effect.color
                symbols[room_idx] = room.effect.symbol
        for p in path:
            pos = p.position
            colors[pos] = "#FFFF00"  # Highlight path 
        colors[path[-1].position] = "#008000"  # Highlight current position in green

        return colors, symbols
                
    def visualize(self, latest_path):
        self.path.append(latest_path)
        colors, symbols= self.getVisualizationAttributes(self.rooms, self.path)

     
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
        
        plt.title(f"A* Solution  \n Step:{len(self.path)-1} | Current Total Cost: {self.path[-1].total_cost:.2f}", fontsize = 30)
        plt.show()
