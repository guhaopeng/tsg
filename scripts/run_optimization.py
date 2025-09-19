import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from scripts.topology_optimization import TopologyOptimization
from scripts.visualize_voxels import visualize_voxels

# 加载密度场
density_field = np.load('d:/TSG/tsg/outputyz_gs10.npy')
print(f"密度场形状: {density_field.shape}")

# 创建优化器实例
optimizer = TopologyOptimization(
    density_field=density_field,
    volume_fraction=0.4,  # 目标体积分数
    penal=3.0,  # SIMP惩罚因子
    rmin=1.5  # 滤波半径
)

# 设置边界条件
# 1. 找到Y值最小的体素作为固定节点
base_mask = density_field > 0.1
min_y = np.min(np.where(base_mask)[1]) if np.any(base_mask) else 0
min_y_mask = base_mask & (np.indices(density_field.shape)[1] == min_y)
min_y_voxels = np.where(min_y_mask)

# 设置固定节点
fixed_nodes = []
for i in range(len(min_y_voxels[0])):
    x = min_y_voxels[0][i]
    y = min_y
    z = min_y_voxels[2][i]
    # 计算节点编号（与有限元分析中的节点编号对应）
    node_id = z + x * density_field.shape[2] + y * density_field.shape[0] * density_field.shape[2]
    fixed_nodes.append(node_id)

# 2. 找到指定范围内且Y值最大的体素（座面区域）作为受力节点
target_mask = base_mask & \
             (np.indices(density_field.shape)[0] >= 0) & (np.indices(density_field.shape)[0] <= 22) & \
             (np.indices(density_field.shape)[1] >= 10) & (np.indices(density_field.shape)[1] <= 20) & \
             (np.indices(density_field.shape)[2] >= 12) & (np.indices(density_field.shape)[2] <= 25)

force_nodes = []
forces = []
if np.any(target_mask):
    max_y = np.max(np.where(target_mask)[1])
    max_y_mask = target_mask & (np.indices(density_field.shape)[1] == max_y)
    max_y_voxels = np.where(max_y_mask)
    
    # 计算每个节点的压力（总压力2N，均匀分布）
    total_force = 2.0
    node_force = -total_force / len(max_y_voxels[0])
    
    # 设置受力节点和对应的力
    for i in range(len(max_y_voxels[0])):
        x = max_y_voxels[0][i]
        y = max_y
        z = max_y_voxels[2][i]
        # 计算节点编号（与有限元分析中的节点编号对应）
        node_id = z + x * density_field.shape[2] + y * density_field.shape[0] * density_field.shape[2]
        force_nodes.append(node_id)
        forces.append((0, node_force, 0))  # 向-y方向的压力

print(f"固定节点数量: {len(fixed_nodes)}")
print(f"受力节点数量: {len(force_nodes)}")

# 设置边界条件
optimizer.set_boundary_conditions(fixed_nodes, force_nodes, forces)

# 运行优化
print("开始拓扑优化...")
compliance_history, density_history = optimizer.optimize(max_iter=120, tolerance=1e-2)

# 保存结果
print("保存优化结果...")
output_dir = os.path.join(project_root, 'output')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'optimized_density.npy')
np.save(output_path, optimizer.density)

# 可视化最终结果
visualize_voxels(optimizer.density, threshold=0.1)
#保存结构图像
plt.savefig(os.path.join(output_dir, 'optimized_structure.png'))
# 关闭当前的figure窗口
plt.close()



print("优化完成！")