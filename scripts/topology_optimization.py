import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve, use_solver

# 配置求解器
use_solver(useUmfpack=False)

class TopologyOptimization:
    def __init__(self, density_field, volume_fraction=0.4, penal=2.0, rmin=1.5):
        """初始化拓扑优化器

        Args:
            density_field (ndarray): 密度场，形状为(nx, ny, nz)
            volume_fraction (float): 目标体积分数
            penal (float): SIMP惩罚因子
            rmin (float): 滤波半径
        """
        # 初始化密度场，使用更大的最小密度值
        self.density = np.maximum(0.02, density_field.copy())  # 增大初始密度
        self.volume_fraction = volume_fraction
        self.penal = penal  # 减小SIMP惩罚因子
        self.rmin = rmin
        
        # 网格尺寸
        self.nx, self.ny, self.nz = density_field.shape
        self.nelx = self.nx - 1
        self.nely = self.ny - 1
        self.nelz = self.nz - 1
        
        # 材料属性
        self.E = 1.0  # 杨氏模量
        self.nu = 0.3  # 泊松比
        
        # 初始化有限元分析所需的变量
        self.initialize_finite_element()
        
        # 初始化边界条件
        self.fixed_dofs = []
        self.force_dofs = []
        self.forces = []
    
    def initialize_finite_element(self):
        """初始化有限元分析所需的变量"""
        # 计算单元刚度矩阵
        self.KE = self._compute_element_stiffness_matrix()
        self.KE_flat = self.KE.flatten()
        
        # 初始化单元到自由度的映射
        self.edofMat = self._compute_element_dof_matrix()
        
        # 初始化单元索引
        self.element_indices = np.arange(self.nelx * self.nely * self.nelz)
    
    def _compute_element_stiffness_matrix(self):
        """计算单元刚度矩阵"""
        k = np.array([
            1/2-self.nu/6,   1/8+self.nu/8,   -1/4-self.nu/12, -1/8+3*self.nu/8,
            -1/4+self.nu/12, -1/8-self.nu/8,    self.nu/6,       1/8-3*self.nu/8
        ])
        
        KE = self.E/(1-self.nu**2) * np.array([
            [k[0], k[1], k[1], k[2], k[4], k[4], k[3], k[3], k[3], k[1], k[1], k[2],
             k[4], k[4], k[3], k[3], k[3], k[1], k[1], k[2], k[4], k[4], k[3], k[3]],
            [k[1], k[0], k[1], k[3], k[3], k[4], k[2], k[4], k[3], k[1], k[0], k[1],
             k[3], k[3], k[4], k[2], k[4], k[3], k[1], k[0], k[1], k[3], k[3], k[4]],
            [k[1], k[1], k[0], k[3], k[4], k[3], k[4], k[2], k[4], k[1], k[1], k[0],
             k[3], k[4], k[3], k[4], k[2], k[4], k[1], k[1], k[0], k[3], k[4], k[3]],
            [k[2], k[3], k[3], k[0], k[5], k[5], k[5], k[5], k[5], k[4], k[4], k[4],
             k[1], k[1], k[1], k[1], k[1], k[1], k[4], k[4], k[4], k[1], k[1], k[1]],
            [k[4], k[3], k[4], k[5], k[0], k[5], k[5], k[5], k[5], k[3], k[4], k[3],
             k[2], k[3], k[2], k[3], k[2], k[3], k[3], k[4], k[3], k[2], k[3], k[2]],
            [k[4], k[4], k[3], k[5], k[5], k[0], k[5], k[5], k[5], k[3], k[3], k[4],
             k[3], k[2], k[3], k[2], k[3], k[2], k[3], k[3], k[4], k[3], k[2], k[3]],
            [k[3], k[2], k[4], k[5], k[5], k[5], k[0], k[5], k[5], k[4], k[3], k[3],
             k[1], k[1], k[1], k[1], k[1], k[1], k[4], k[3], k[3], k[1], k[1], k[1]],
            [k[3], k[4], k[2], k[5], k[5], k[5], k[5], k[0], k[5], k[3], k[4], k[3],
             k[2], k[3], k[2], k[3], k[2], k[3], k[3], k[4], k[3], k[2], k[3], k[2]],
            [k[3], k[3], k[4], k[5], k[5], k[5], k[5], k[5], k[0], k[3], k[3], k[4],
             k[3], k[2], k[3], k[2], k[3], k[2], k[3], k[3], k[4], k[3], k[2], k[3]],
            [k[1], k[1], k[1], k[4], k[3], k[3], k[4], k[3], k[3], k[0], k[1], k[1],
             k[2], k[4], k[4], k[3], k[3], k[3], k[1], k[1], k[1], k[4], k[3], k[3]],
            [k[1], k[0], k[1], k[4], k[4], k[3], k[3], k[4], k[3], k[1], k[0], k[1],
             k[3], k[3], k[4], k[2], k[4], k[3], k[1], k[0], k[1], k[3], k[3], k[4]],
            [k[2], k[1], k[0], k[4], k[3], k[4], k[3], k[3], k[4], k[1], k[1], k[0],
             k[3], k[4], k[3], k[4], k[2], k[4], k[1], k[1], k[0], k[3], k[4], k[3]],
            [k[4], k[3], k[3], k[1], k[2], k[3], k[1], k[2], k[3], k[2], k[3], k[3],
             k[0], k[5], k[5], k[5], k[5], k[5], k[4], k[4], k[4], k[1], k[1], k[1]],
            [k[4], k[3], k[4], k[1], k[3], k[2], k[1], k[3], k[2], k[4], k[3], k[4],
             k[5], k[0], k[5], k[5], k[5], k[5], k[3], k[4], k[3], k[2], k[3], k[2]],
            [k[3], k[4], k[3], k[1], k[2], k[3], k[1], k[2], k[3], k[4], k[4], k[3],
             k[5], k[5], k[0], k[5], k[5], k[5], k[3], k[3], k[4], k[3], k[2], k[3]],
            [k[3], k[2], k[4], k[1], k[3], k[2], k[1], k[3], k[2], k[3], k[2], k[4],
             k[5], k[5], k[5], k[0], k[5], k[5], k[4], k[3], k[3], k[1], k[1], k[1]],
            [k[3], k[4], k[2], k[1], k[2], k[3], k[1], k[2], k[3], k[3], k[4], k[2],
             k[5], k[5], k[5], k[5], k[0], k[5], k[3], k[4], k[3], k[2], k[3], k[2]],
            [k[3], k[3], k[4], k[1], k[3], k[2], k[1], k[3], k[2], k[3], k[3], k[4],
             k[5], k[5], k[5], k[5], k[5], k[0], k[3], k[3], k[4], k[3], k[2], k[3]],
            [k[1], k[1], k[1], k[4], k[3], k[3], k[4], k[3], k[3], k[1], k[1], k[1],
             k[4], k[3], k[3], k[4], k[3], k[3], k[0], k[1], k[1], k[2], k[4], k[4]],
            [k[1], k[0], k[1], k[4], k[4], k[3], k[3], k[4], k[3], k[1], k[0], k[1],
             k[4], k[4], k[3], k[3], k[4], k[3], k[1], k[0], k[1], k[3], k[3], k[4]],
            [k[2], k[1], k[0], k[4], k[3], k[4], k[3], k[3], k[4], k[1], k[1], k[0],
             k[4], k[3], k[4], k[3], k[3], k[4], k[1], k[1], k[0], k[3], k[4], k[3]],
            [k[4], k[3], k[3], k[1], k[2], k[3], k[1], k[2], k[3], k[4], k[3], k[3],
             k[1], k[2], k[3], k[1], k[2], k[3], k[2], k[3], k[3], k[0], k[5], k[5]],
            [k[4], k[3], k[4], k[1], k[3], k[2], k[1], k[3], k[2], k[3], k[3], k[4],
             k[1], k[3], k[2], k[1], k[3], k[2], k[4], k[3], k[4], k[5], k[0], k[5]],
            [k[3], k[4], k[3], k[1], k[2], k[3], k[1], k[2], k[3], k[3], k[4], k[3],
             k[1], k[2], k[3], k[1], k[2], k[3], k[4], k[4], k[3], k[5], k[5], k[0]]
        ])
        
        return KE

    
    def _compute_element_dof_matrix(self):
        """计算单元自由度矩阵"""
        # 计算节点编号
        nodenrs = np.arange((self.nelx+1)*(self.nely+1)*(self.nelz+1)).reshape(self.nelx+1, self.nely+1, self.nelz+1)
        

        # 初始化单元自由度矩阵  
        edofMat = np.zeros((self.nelx*self.nely*self.nelz, 24), dtype=int)
        
        for elz in range(self.nelz):
            for ely in range(self.nely):
                for elx in range(self.nelx):
                    # 计算单元的8个节点编号
                    n1 = nodenrs[elx, ely, elz]
                    n2 = nodenrs[elx+1, ely, elz]
                    n3 = nodenrs[elx+1, ely+1, elz]
                    n4 = nodenrs[elx, ely+1, elz]
                    n5 = nodenrs[elx, ely, elz+1]
                    n6 = nodenrs[elx+1, ely, elz+1]
                    n7 = nodenrs[elx+1, ely+1, elz+1]
                    n8 = nodenrs[elx, ely+1, elz+1]
                    
                    # 计算单元编号
                    el = elz*self.nelx*self.nely + ely*self.nelx + elx
                    
                    # 设置单元的24个自由度（每个节点3个自由度）
                    edofMat[el] = np.array([
                        3*n1, 3*n1+1, 3*n1+2,
                        3*n2, 3*n2+1, 3*n2+2,
                        3*n3, 3*n3+1, 3*n3+2,
                        3*n4, 3*n4+1, 3*n4+2,
                        3*n5, 3*n5+1, 3*n5+2,
                        3*n6, 3*n6+1, 3*n6+2,
                        3*n7, 3*n7+1, 3*n7+2,
                        3*n8, 3*n8+1, 3*n8+2
                    ])
        
        return edofMat
    
    def set_boundary_conditions(self, fixed_nodes, force_nodes, forces):
        """设置边界条件

        Args:
            fixed_nodes (list): 固定节点的坐标列表，每个元素为节点编号
            force_nodes (list): 受力节点的坐标列表，每个元素为节点编号
            forces (list): 对应的力向量列表，每个元素为(fx, fy, fz)
        """
        # 清空原有的边界条件
        self.fixed_dofs = []
        self.force_dofs = []
        self.forces = []
        
        # 设置固定自由度
        for node in fixed_nodes:
            # 将节点的三个自由度都固定
            self.fixed_dofs.extend([3*node, 3*node+1, 3*node+2])
        
        # 设置受力自由度和对应的力
        for node, force in zip(force_nodes, forces):
            fx, fy, fz = force
            if fx != 0:
                self.force_dofs.append(3*node)
                self.forces.append(fx)
            if fy != 0:
                self.force_dofs.append(3*node+1)
                self.forces.append(fy)
            if fz != 0:
                self.force_dofs.append(3*node+2)
                self.forces.append(fz)
    
    def finite_element_analysis(self):
        """有限元分析"""
        # 计算每个单元的密度因子
        density_factors = np.maximum(0.02, self.density[:-1, :-1, :-1].reshape(-1))  # 增大最小密度
        density_factors = density_factors ** self.penal  # 应用SIMP惩罚
        
        # 组装全局刚度矩阵
        ndof = (self.nelx+1)*(self.nely+1)*(self.nelz+1)*3  # 总自由度数
        
        # 分批处理以节省内存
        batch_size = 200  # 每批处理的单元数
        num_batches = (len(density_factors) + batch_size - 1) // batch_size
        
        # 初始化全局刚度矩阵
        K = None
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(density_factors))
            
            # 初始化当前批次的COO矩阵数据
            iK = np.zeros(24*24*(end_idx-start_idx), dtype=int)
            jK = np.zeros(24*24*(end_idx-start_idx), dtype=int)
            sK = np.zeros(24*24*(end_idx-start_idx))
            
            # 组装当前批次的刚度矩阵
            for i, factor in enumerate(density_factors[start_idx:end_idx]):
                # 获取当前单元的自由度编号
                dofs = self.edofMat[start_idx+i]
                
                # 生成网格索引
                I, J = np.meshgrid(dofs, dofs)
                
                # 计算当前位置
                idx = i*576
                
                # 存储行列索引和数据
                iK[idx:idx+576] = I.flatten()
                jK[idx:idx+576] = J.flatten()
                sK[idx:idx+576] = factor * self.KE_flat
            
            # 创建当前批次的刚度矩阵
            K_batch = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
            
            # 累加到全局刚度矩阵
            if K is None:
                K = K_batch
            else:
                K = K + K_batch
            
            # 释放内存
            del iK, jK, sK, K_batch
        
        # 添加正则化项
        K = K + diags(np.ones(ndof) * 1e-3)  # 增大正则化系数
        
        # 准备力向量
        F = np.zeros(ndof)
        F[self.force_dofs] = self.forces
        
        # 准备自由度
        free_dofs = np.setdiff1d(np.arange(ndof), self.fixed_dofs)
        
        # 求解位移
        U = np.zeros(ndof)
        try:
            # 提取自由自由度对应的矩阵和向量
            K_free = K[free_dofs][:, free_dofs]
            F_free = F[free_dofs]
            
            # 使用直接求解器
            U_free = spsolve(K_free, F_free)
            U[free_dofs] = U_free
            
            # 计算柔度（应变能）
            self.compliance = float(U.T @ K @ U)  # 确保转换为标量
            
        except Exception as e:
            print(f"求解器错误: {str(e)}")
            raise
        
        self.displacements = U
    
    def optimize(self, max_iter=50, tolerance=1e-3):
        """运行拓扑优化

        Args:
            max_iter (int): 最大迭代次数
            tolerance (float): 收敛容差

        Returns:
            tuple: (compliance_history, density_history)
        """
        compliance_history = []
        density_history = []
        old_compliance = 0
        
        for iteration in range(max_iter):
            # 有限元分析
            self.finite_element_analysis()
            
            # 检查数值稳定性
            if not np.isfinite(self.compliance) or self.compliance < 0:
                print("警告：检测到数值不稳定，终止优化")
                break
            
            # 记录历史数据
            compliance_history.append(self.compliance)
            density_history.append(self.density.copy())
            
            # 计算当前体积分数
            volume_fraction = float(np.mean(self.density))  # 确保转换为标量
            
            # 打印当前迭代信息
            print(f"第{iteration+1}步迭代：柔度={self.compliance:.4f}，体积分数={volume_fraction:.4f}")
            
            # 检查收敛性
            if iteration > 0:
                change = abs(self.compliance - old_compliance) / (abs(self.compliance) + 1e-9)
                if change < tolerance:
                    print("优化已收敛！")
                    break
            
            old_compliance = self.compliance
            
            # 更新设计变量（密度场）
            self._update_density()
        
        return compliance_history, density_history
    
    def _update_density(self):
        """更新密度场"""
        # 计算灵敏度
        sensitivity = self._compute_sensitivity()
        
        # 应用滤波
        filtered_sensitivity = self._apply_filter(sensitivity)
        
        # 更新密度场
        l1 = 0
        l2 = 1e9
        move = 0.05  # 减小移动限制
        
        while abs(l2 - l1) > 1e-4:
            lmid = 0.5 * (l2 + l1)
            
            # 计算新的密度场
            # 确保灵敏度为负值（最小化问题）
            filtered_sensitivity = np.minimum(0, filtered_sensitivity)
            
            # 计算更新因子，避免数值不稳定
            update_factor = np.zeros_like(filtered_sensitivity)
            valid_mask = filtered_sensitivity < 0
            update_factor[valid_mask] = np.sqrt(-filtered_sensitivity[valid_mask] / (lmid + 1e-6))
            update_factor = np.maximum(0.02, update_factor)  # 增大最小更新因子
            
            # 应用移动限制
            new_density = np.clip(
                self.density * update_factor,
                np.maximum(0.02, self.density - move),  # 增大最小密度
                np.minimum(1.0, self.density + move)
            )
            
            # 计算体积约束
            volume = float(np.mean(new_density))  # 确保转换为标量
            
            # 更新拉格朗日乘子
            if volume > self.volume_fraction:
                l1 = lmid
            else:
                l2 = lmid
        
        self.density = new_density
    
    def _compute_sensitivity(self):
        """计算灵敏度"""
        # 重塑位移场
        U = self.displacements.reshape(-1, 3)
        
        # 初始化灵敏度场
        sensitivity = np.zeros_like(self.density)
        
        # 计算每个单元的灵敏度
        for elz in range(self.nelz):
            for ely in range(self.nely):
                for elx in range(self.nelx):
                    # 获取单元节点位移
                    el = elz*self.nelx*self.nely + ely*self.nelx + elx
                    Ue = self.displacements[self.edofMat[el]]
                    
                    # 计算单元灵敏度
                    sensitivity[elx, ely, elz] = -self.penal * (self.density[elx, ely, elz] ** (self.penal-1)) * Ue @ (self.KE @ Ue)
        
        return sensitivity
    
    def _apply_filter(self, field):
        """应用密度滤波

        Args:
            field (ndarray): 需要滤波的场

        Returns:
            ndarray: 滤波后的场
        """
        # 创建滤波核
        rmin = int(np.ceil(self.rmin))
        [x, y, z] = np.ogrid[-rmin:rmin+1, -rmin:rmin+1, -rmin:rmin+1]
        kernel = ((rmin - np.sqrt(x*x + y*y + z*z)) > 0).astype(float)
        
        # 手动实现3D卷积
        filtered_field = np.zeros_like(field)
        padded_field = np.pad(field, rmin, mode='reflect')
        padded_density = np.pad(self.density, rmin, mode='reflect')
        
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                for k in range(field.shape[2]):
                    # 提取局部区域
                    field_window = padded_field[i:i+2*rmin+1, j:j+2*rmin+1, k:k+2*rmin+1]
                    density_window = padded_density[i:i+2*rmin+1, j:j+2*rmin+1, k:k+2*rmin+1]
                    
                    # 计算加权和
                    numerator = np.sum(kernel * field_window * density_window)
                    denominator = np.sum(kernel * density_window)
                    
                    # 更新滤波后的值
                    filtered_field[i, j, k] = numerator / (denominator + 1e-6)
        
        return filtered_field