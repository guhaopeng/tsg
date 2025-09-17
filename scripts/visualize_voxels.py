import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path

def visualize_voxels(voxel_path, output_path=None, alpha=0.3, threshold=0.0, color_by_density=False):
    """
    可视化体素数据
    
    Args:
        voxel_path: NPY文件路径
        output_path: 输出图像路径（可选）
        alpha: 体素透明度
        threshold: 密度阈值，只显示大于此值的体素
        color_by_density: 是否根据密度值着色
    """
    print(f"正在加载体素数据: {voxel_path}")
    density = np.load(voxel_path)
    
    print(f"体素数据形状: {density.shape}")
    print(f"密度值范围: [{density.min():.3f}, {density.max():.3f}]")
    
    # 创建直方图显示密度分布
    #plt.figure(figsize=(10, 5))
    #plt.hist(density.ravel(), bins=50, range=(0, 1))
    #plt.title("密度值分布直方图")
    #plt.xlabel("密度值")
    #plt.ylabel("频率")
    #if output_path:
     #   hist_path = str(Path(output_path).parent / f"{Path(output_path).stem}_hist.png")
      #  plt.savefig(hist_path, dpi=300, bbox_inches='tight')
   # plt.close()
    
    # 应用阈值
    mask = density > threshold
    print(f"应用阈值{threshold}后的非零体素数量: {np.count_nonzero(mask)}")
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    if color_by_density:
        # 根据密度值设置颜色
        colors = np.zeros(density.shape + (4,))  # RGBA颜色数组
        normalized_density = (density - density.min()) / (density.max() - density.min())
        colors[mask] = plt.cm.viridis(normalized_density[mask])
        colors[mask, 3] = alpha  # 设置透明度
    else:
        # 使用固定的黄色
        colors = np.zeros(density.shape + (4,))
        colors[mask] = [1.0, 1.0, 0.0, alpha]
    
    # 绘制体素
    ax.voxels(mask,
              facecolors=colors,
              edgecolor='gray',
              linewidth=0.3)
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置轴范围
    ax.set_xlim([0, density.shape[0]])
    ax.set_ylim([0, density.shape[1]])
    ax.set_zlim([0, density.shape[2]])
    
    # 设置标题
    plt.title(f'3D体素可视化 (阈值={threshold})')
    
    # 调整布局
    plt.tight_layout()
    
    if output_path:
        # 保存图像
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n可视化结果已保存至: {output_path}")
    
    # 显示图像
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="可视化体素数据")
    parser.add_argument("--voxel-path", type=str, required=True, help="输入的体素数据文件路径(.npy)")
    parser.add_argument("--output-path", type=str, default=None, help="输出图像路径(.png/.jpg)")
    parser.add_argument("--alpha", type=float, default=0.3, help="体素透明度(0-1)")
    parser.add_argument("--threshold", type=float, default=0.5, help="密度阈值(0-1)，只显示大于此值的体素")
    parser.add_argument("--color-by-density", action="store_true", help="是否根据密度值着色")
    
    args = parser.parse_args()
    
    # 如果未指定输出路径，则根据输入文件名生成
    if args.output_path is None:
        input_path = Path(args.voxel_path)
        args.output_path = str(input_path.parent / f"{input_path.stem}_vis.png")
    
    try:
        visualize_voxels(
            args.voxel_path,
            args.output_path,
            args.alpha,
            args.threshold,
            args.color_by_density
        )
        print("\n可视化完成!")
        
    except Exception as e:
        print(f"\n可视化过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()