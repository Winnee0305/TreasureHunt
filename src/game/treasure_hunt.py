from src.models.hex import plot_hex_maze
import matplotlib.pyplot as plt

class Effect: # Represents an effect in the hexagonal grid
    def __init__(self, name):
        self.name = name # Name of the effect
        self.color = None # Color associated with the effect
        self.symbol = None # Symbol associated with the effect
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
        self.room_idx = room_idx # Unique identifier for the room, cordinates (row, col)
        self.parent = parent # Parent room in the path
        self.neighbors = [] # List of neighboring rooms
        self.effect = Effect('None')  # Default effect

    def setEffect(self, effect): # Set the effect for the room
        self.effect = Effect(effect) # Initialize the effect based on its name
 
class TreasureHunt: # Represents the treasure hunt game in a hexagonal grid
    def __init__(self, nrow, ncol):
        self.nrow = nrow # Number of rows in the hexagonal grid
        self.ncol = ncol # Number of columns in the hexagonal grid
        self.rooms = {} # Dictionary to hold rooms, indexed by their coordinates (row, col)
        self.create_rooms()
        self.starting_room = (0, 0)  # Starting room coordinates
        self.path = [] # List to hold the path taken during the treasure hunt

    def create_rooms(self): # Create rooms in the hexagonal grid based on the number of rows and columns
        for row in range(self.nrow):
            for col in range(self.ncol):
                room_idx = (row, col)
                self.rooms[room_idx] = HexRoom(room_idx)

    def setEffect(self, effect): # Set effects for specific rooms based on the provided dictionary
        for room_idx, effect_value in effect.items():
            if room_idx in self.rooms:
                self.rooms[room_idx].setEffect(effect_value)

    def getVisualizationAttributes(self, rooms, path): # Get visualization attributes for the rooms and path
        colors = {}
        symbols = {}
        for room_idx, room in rooms.items(): # Iterate through each room
            if room.effect is not None: # Check if the room has an effect
                colors[room_idx] = room.effect.color # Set the color for the room
                symbols[room_idx] = room.effect.symbol # Set the symbol for the room
        for p in path: # Iterate through the path taken
            pos = p.position # Get the position of the room in the path
            colors[pos] = "#FFFF00"  # Highlight path 
        colors[path[-1].position] = "#008000"  # Highlight current position in green

        return colors, symbols
                
    def visualize(self, latest_path): # Visualize the current state of the treasure hunt
        self.path.append(latest_path) # Append the latest path to the existing path
        colors, symbols= self.getVisualizationAttributes(self.rooms, self.path) # Get colors and symbols for visualization

        fig, ax = plot_hex_maze(self.nrow, self.ncol, hex_size=1, # Plot the hexagonal grid
                        colors=colors,
                        symbols=symbols)
        
        plt.title(f"A* Solution  \n Step:{len(self.path)-1} | Current Total Cost: {self.path[-1].total_cost:.2f}", fontsize = 30) # Set the title of the plot
        plt.show()
