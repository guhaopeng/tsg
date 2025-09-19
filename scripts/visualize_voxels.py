import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_voxels(density, threshold=0.5):
    """
    Visualize 3D voxel data using matplotlib.
    
    Args:
        density: 3D numpy array containing density values
        threshold: Density threshold for visualization (default: 0.5)
    """
    # Convert density values to boolean based on threshold
    voxels = density > threshold
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels
    colors = np.zeros(voxels.shape + (4,))
    colors[voxels] = [0.5, 0.5, 0.8, 0.7]  # Semi-transparent blue for solid voxels
    ax.voxels(voxels, facecolors=colors)
    
    # Set labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title(f'3D结构 (η>{threshold})')
    
    # Show plot
    plt.show()