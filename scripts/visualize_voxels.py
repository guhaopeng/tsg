import matplotlib.pyplot as plt
import numpy as np
import os
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

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot before showing
    save_path = os.path.join(output_dir, 'optimized_structure.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"结构图已保存至: {save_path}")
    plt.show()
    
    # 创建三个面的投影视图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
    
    # XY平面视图（俯视图）
    ax1.voxels(voxels, facecolors=colors)
    ax1.view_init(90, -90)  # 俯视图
    ax1.set_title('XY平面视图（俯视图）')
    ax1.axis('off')
    
    # YZ平面视图（左视图）
    ax2.voxels(voxels, facecolors=colors)
    ax2.view_init(0, 0)  # 左视图
    ax2.set_title('XY平面视图（左视图）')
    ax2.axis('off')
    
    # XZ平面视图（正视图，逆时针旋转90度）
    ax3.voxels(voxels, facecolors=colors)
    ax3.view_init(0, 45)  # 正视图
    ax3.set_title('XZ平面视图（正视图）')
    ax3.axis('off')
    
    plt.tight_layout()
    # 保存投影视图
    proj_save_path = os.path.join(output_dir, 'optimized_structure_projections.png')
    plt.savefig(proj_save_path, dpi=300, bbox_inches='tight')
    print(f"投影视图已保存至: {proj_save_path}")
    plt.show()

if __name__ == "__main__":
    # 加载示例数据进行测试
    try:
        density = np.load("outputyz_latent0.5.npy")
        visualize_voxels(density, threshold=0.5)
    except FileNotFoundError:
        print("未找到优化后的密度场数据文件。")