import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as colors

def create_arrow(ax, start, direction, length=2, color='b', alpha=0.8):
    ax.quiver(start[0], start[1], start[2],
              direction[0], direction[1], direction[2],
              length=length, color=color, alpha=alpha,
              arrow_length_ratio=0.3)

def visualize_constraints(density_path):
    # 加载密度场数据
    density = np.load(density_path)
    print(f"体素数据形状: {density.shape}")
    
    # 创建图形
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建基础体素掩码
    base_mask = density > 0.0
    
    # 找到Y值最小的体素
    min_y = np.min(np.where(base_mask)[1]) if np.any(base_mask) else 0
    min_y_mask = base_mask & (np.indices(density.shape)[1] == min_y)
    min_y_voxels = np.where(min_y_mask)
    print(f'找到{len(min_y_voxels[0])}个Y={min_y}的体素点')
    
    # 输出最小Y值平面体素的节点约束信息
    print('\n固定约束节点（UX=UY=UZ=0）：')
    for i in range(len(min_y_voxels[0])):
        x_coord = min_y_voxels[0][i]
        y_coord = min_y
        z_coord = min_y_voxels[2][i]
        print(f'节点坐标：({x_coord}, {y_coord}, {z_coord})')
    
    # 找到指定范围内且Y值最大的体素（座面区域）
    target_mask = base_mask & \
                 (np.indices(density.shape)[0] >= 0) & (np.indices(density.shape)[0] <= 22) & \
                 (np.indices(density.shape)[1] >= 10) & (np.indices(density.shape)[1] <= 20) & \
                 (np.indices(density.shape)[2] >= 12) & (np.indices(density.shape)[2] <= 25)
    
    if np.any(target_mask):
        max_y = np.max(np.where(target_mask)[1])
        max_y_mask = target_mask & (np.indices(density.shape)[1] == max_y)
        max_y_voxels = np.where(max_y_mask)
        print(f'\n找到{len(max_y_voxels[0])}个Y={max_y}的座面体素点')
        
        # 输出座面体素的节点信息
        print('座面体素节点（施加-y方向压力）：')
        total_force = 1.0  # 总压力为2N
        node_force = -total_force / len(max_y_voxels[0])  # 每个节点的压力
        print(f'每个节点压力：{abs(node_force)}N（-y方向）')
        
        for i in range(len(max_y_voxels[0])):
            x_coord = max_y_voxels[0][i]
            y_coord = max_y
            z_coord = max_y_voxels[2][i]
            print(f'节点坐标：({x_coord}, {y_coord}, {z_coord})')
    
    # 创建颜色数组
    colors_array = np.zeros(density.shape + (4,))  # RGBA颜色数组
    
    # 设置基础体素的颜色（灰色半透明）
    colors_array[base_mask] = [0.7, 0.7, 0.7, 0.3]
    
    # 设置最小Y值体素的颜色（黄色）
    colors_array[min_y_mask] = [1.0, 1.0, 0.0, 0.8]
    
    # 设置最大Y值体素的颜色（红色）
    if np.any(target_mask):
        colors_array[max_y_mask] = [1.0, 0.0, 0.0, 1.0]
    
    # 绘制所有体素
    ax.voxels(base_mask,
              facecolors=colors_array,
              edgecolor='gray',
              linewidth=0.3)
    
    # 添加固定约束（粉色平面）
    x_range = np.arange(0, 32)
    z_range = np.arange(0, 32)
    X, Z = np.meshgrid(x_range, z_range)
    Y = np.full_like(X, min_y)
    ax.plot_surface(X, Y, Z, color='pink', alpha=0.2)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 在座面上添加向-y方向的压力箭头
    if np.any(target_mask):
        # 找到每个座面体素的位置
        for i in range(len(max_y_voxels[0])):
            x = max_y_voxels[0][i]
            y = max_y
            z = max_y_voxels[2][i]
            # 在所有座面体素上添加箭头
            create_arrow(ax, [x, y, z], [0, -1, 0], length=1, color='blue')
    
    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('sensitivity and constraints')
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, alpha=0.3, label='density field'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='yellow', markersize=10, alpha=0.8, label=f'Y={min_y} plane voxels'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, alpha=1.0, label=f'co-planar voxels (Y={max_y if "max_y" in locals() else "N/A"})'),
        Line2D([0], [0], color='pink', alpha=0.2, label='fixed constraints'),
        Line2D([0], [0], color='blue', alpha=0.8, label='loads')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 保存图像
    output_path = density_path.replace('.npy', '_constraints.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    density_path = 'd:/TSG/tsg/outputyz_steps20.npy'
    visualize_constraints(density_path)