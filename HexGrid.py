import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon

def create_hex_grid(rows, cols, hex_size=1, colors=None, labels=None, symbols=None):
    """
    Create a hexagonal grid visualization.
    
    Parameters:
    -----------
    rows : int
        Number of rows in the grid
    cols : int
        Number of columns in the grid
    hex_size : float
        Size of the hexagons (radius)
    colors : dict
        Dictionary with (row, col) tuples as keys and colors as values
    labels : dict
        Dictionary with (row, col) tuples as keys and label texts as values
    symbols : dict
        Dictionary with (row, col) tuples as keys and symbol characters as values
    """
    # Initialize figure with white background and larger size
    fig, ax = plt.subplots(figsize=(20, 15), facecolor='white')
    ax.set_facecolor('white')
    
    # Set default dictionaries if None
    colors = colors or {}
    labels = labels or {}
    symbols = symbols or {}
    
    # Constants for hex grid geometry
    spacing_factor = 1.1  # Increase this to add more space between hexagons
    hex_width = 2 * hex_size
    hex_height = np.sqrt(3) * hex_size
    
    # Calculate coordinates
    for row in range(rows):
        for col in range(cols):
            # Calculate base position with spacing
            x = col * (3/4) * hex_width * spacing_factor
            y = row * hex_height * spacing_factor
            
            # Offset odd-numbered columns up
            if col % 2:
                y += hex_height * spacing_factor / 2
            
            # Create hexagon with thicker edge
            color = colors.get((row, col), 'white')  # Set default color to white
            hex = RegularPolygon((x, y),
                               numVertices=6,
                               radius=hex_size * 0.95,  # Make hexagons slightly smaller
                               orientation=np.pi/2,  # No rotation
                               facecolor=color,
                               edgecolor='black',
                               linewidth=1.5)
            ax.add_patch(hex)
            
            # Add symbol if specified
            if (row, col) in symbols:
                plt.text(x, y, symbols[(row, col)],
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=40,
                        color='black')
    
    # Calculate proper limits
    width = cols * (2/3) * hex_width * spacing_factor + hex_width/4
    height = rows * hex_height 
    
    # Set aspect ratio to equal and set limits with padding
    ax.set_aspect('equal')
    padding = hex_size * 1.5
    
    ax.set_xlim(-padding, width + padding)
    ax.set_ylim(-padding, height + padding)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout to prevent cutting
    plt.tight_layout(pad=1.5)
    
    return fig, ax

