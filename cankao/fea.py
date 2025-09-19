import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from tqdm import tqdm

class FEAnalysis:
    """增强的有限元分析模块"""
    def __init__(self, young_modulus=1.0, poisson_ratio=0.3):
        self.E = young_modulus   # 杨氏模量
        self.v = poisson_ratio   # 泊松比 
        self.D = self._compute_constitutive_matrix()
        self.KE = self._element_stiffness_matrix()
        
    def _compute_constitutive_matrix(self):
        """计算单位本构矩阵(3D线性弹性)"""
        v = self.v
        D = np.array([
            [1-v, v, v, 0, 0, 0],
            [v, 1-v, v, 0, 0, 0],
            [v, v, 1-v, 0, 0, 0],
            [0, 0, 0, (1-2*v)/2, 0, 0],
            [0, 0, 0, 0, (1-2*v)/2, 0],
            [0, 0, 0, 0, 0, (1-2*v)/2]
        ]) * (self.E / ((1+v)*(1-2*v)))
        return D
    
    def _element_stiffness_matrix(self):
        """计算3D 8节点六面体单元刚度矩阵"""
        # 高斯积分点(2x2x2)
        gauss_points = [
            (-0.57735, -0.57735, -0.57735),
            (0.57735, -0.57735, -0.57735),
            (0.57735, 0.57735, -0.57735),
            (-0.57735, 0.57735, -0.57735),
            (-0.57735, -0.57735, 0.57735),
            (0.57735, -0.57735, 0.57735),
            (0.57735, 0.57735, 0.57735),
            (-0.57735, 0.57735, 0.57735)
        ]
        weights = [1.0] * 8
        
        KE = np.zeros((24, 24))
        
        for gp, w in zip(gauss_points, weights):
            xi, eta, zeta = gp
            # 计算形函数导数
            dN = np.array([
                [-(1-eta)*(1-zeta), (1-eta)*(1-zeta), (1+eta)*(1-zeta), -(1+eta)*(1-zeta),
                 -(1-eta)*(1+zeta), (1-eta)*(1+zeta), (1+eta)*(1+zeta), -(1+eta)*(1+zeta)],
                [-(1-xi)*(1-zeta), -(1+xi)*(1-zeta), (1+xi)*(1-zeta), (1-xi)*(1-zeta),
                 -(1-xi)*(1+zeta), -(1+xi)*(1+zeta), (1+xi)*(1+zeta), (1-xi)*(1+zeta)],
                [-(1-xi)*(1-eta), -(1+xi)*(1-eta), -(1+xi)*(1+eta), -(1-xi)*(1+eta),
                 (1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)]
            ]) / 8.0
            
            # 雅可比矩阵
            J = dN @ np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
            ])
            invJ = np.linalg.inv(J)  # 计算逆矩阵
            detJ = np.linalg.det(J)   # 计算行列式

            
            # 应变-位移矩阵B
            B = np.zeros((6, 24))
            for i in range(8):
                dN_xyz = invJ @ dN[:,i]  #dN[:,i] 选取所有行 第i列  

                B[:, 3*i:3*i+3] = [
                    [dN_xyz[0], 0, 0],
                    [0, dN_xyz[1], 0],
                    [0, 0, dN_xyz[2]],
                    [dN_xyz[1], dN_xyz[0], 0],
                    [0, dN_xyz[2], dN_xyz[1]],
                    [dN_xyz[2], 0, dN_xyz[0]]
                ]
            
            KE += B.T @ self.D @ B * detJ * w
        
        return KE
    
    def compute_stiffness_matrix(self, voxel_grid, penal=3.0, E_min=1e-6):
        """计算全局刚度矩阵(考虑SIMP材料插值)"""
        nelx, nely, nelz = voxel_grid.shape
        n_elements = nelx * nely * nelz        # 单元数量
        n_nodes = (nelx+1)*(nely+1)*(nelz+1)   # 节点数量
        dof = 3 * n_nodes                      # 自由度数量
        
        # 材料插值
        E = E_min + voxel_grid**penal * (self.E - E_min)  
        #simp材料插值 E = E_min + p^penal * (E - E_min)
        '''
    组装过程：
        计算网格尺寸和自由度总数
        对每个单元：
          计算8个节点的全局编号
          确定24个自由度(每个节点3个方向)
          将单元刚度矩阵乘以材料系数后添加到全局矩阵
        '''
        
        # 组装全局刚度矩阵(使用COO格式提高效率)
        row_indices = []
        col_indices = []
        data = []
        
        # 确保E是3D数组
        if not isinstance(E, np.ndarray) or len(E.shape) != 3:
            raise ValueError(f"Expected 3D array for E, got shape {E.shape if isinstance(E, np.ndarray) else type(E)}")
        
        # 打印调试信息
        print(f"\nAssembling stiffness matrix with:")
        print(f"  Grid size: {nelx}x{nely}x{nelz}")
        print(f"  E shape: {E.shape}")
        print(f"  E range: [{np.min(E)}, {np.max(E)}]")
        
        for elx in tqdm(range(nelx), desc="Assembling stiffness matrix组装刚度矩阵"):   # 组装刚度矩阵
            for ely in range(nely):
                for elz in range(nelz):
                    # 8个节点的全局索引   
                    n1 = (nely+1)*(nelz+1)*elx + (nelz+1)*ely + elz  # 节点1的全局编号
                    n2 = n1 + 1 
                    n3 = n1 + (nelz+1)
                    n4 = n3 + 1
                    n5 = n1 + (nely+1)*(nelz+1)
                    n6 = n5 + 1
                    n7 = n5 + (nelz+1)
                    n8 = n7 + 1
                    
                    # 24个自由度
                    edof = np.array([
                        3*n1, 3*n1+1, 3*n1+2,
                        3*n2, 3*n2+1, 3*n2+2,
                        3*n3, 3*n3+1, 3*n3+2,
                        3*n4, 3*n4+1, 3*n4+2,
                        3*n5, 3*n5+1, 3*n5+2,
                        3*n6, 3*n6+1, 3*n6+2,
                        3*n7, 3*n7+1, 3*n7+2,
                        3*n8, 3*n8+1, 3*n8+2
                    ])
                    
                    # 获取当前单元的材料属性
                    try:
                        E_el = float(E[elx,ely,elz])
                    except Exception as e:
                        print(f"Error accessing E[{elx},{ely},{elz}]: {str(e)}")
                        print(f"E shape: {E.shape}")
                        raise
                    
                    # 添加单元贡献
                    for i in range(24):  #KE 24*24
                        for j in range(24):
                            row_indices.append(edof[i])
                            col_indices.append(edof[j])
                            data.append(E_el * self.KE[i,j])  #实际的单元刚度矩阵  # 材料属性×几何刚度
        
        # 创建稀疏矩阵
        try:
            K = sp.coo_matrix((data, (row_indices, col_indices)), shape=(dof, dof))
        except Exception as e:
            print(f"Error creating sparse matrix:")
            print(f"  data length: {len(data)}")
            print(f"  row_indices length: {len(row_indices)}")
            print(f"  col_indices length: {len(col_indices)}")
            print(f"  dof: {dof}")
            raise
        return K.tocsc()
    
    def solve(self, K, F, fixed_dofs):
        """求解位移场，使用预处理的迭代求解器"""
        free_dofs = np.setdiff1d(np.arange(K.shape[0]), fixed_dofs)  # 获取所有未被固定的自由度索引(需求解)
        K_free = K[free_dofs,:][:,free_dofs]  # 自由节点的刚度矩阵
        F_free = F[free_dofs]   # 从全局载荷向量 F 中提取自由自由度对应的载荷子向量 

        # 添加数值稳定性处理
        K_free = K_free + sp.eye(K_free.shape[0]) * 1e-8  # 添加小量到对角线

        # 使用不完全LU分解作为预处理器
        try:
            # 增加fill_factor以提高预处理效果
            ilu = spla.spilu(K_free, fill_factor=20)
            M = spla.LinearOperator(K_free.shape, ilu.solve)

            # 调整GMRES参数
            U_free, info = spla.gmres(K_free, F_free, 
                                     M=M, 
                                     tol=1e-6,  # 放宽收敛容差
                                     maxiter=2000,  # 增加最大迭代次数
                                     restart=100)  # 添加重启参数
            
            if info != 0:
                print(f"GMRES failed to converge (info={info}), trying direct solver...")
                U_free = spla.spsolve(K_free, F_free)
                
        except Exception as e:
            print(f"Solver error: {str(e)}, using direct solver...")
            U_free = spla.spsolve(K_free, F_free)
        
        U = np.zeros(K.shape[0])  
        U[free_dofs] = U_free
        return U
    
    def compute_compliance(self, U, K):
        """计算柔度"""
        return U.T @ K @ U
    
    def compute_element_compliance(self, elx, ely, elz, U, nelx=32, nely=32, nelz=32):
        """计算单元柔度贡献
        Args:
            elx, ely, elz: 单元在x,y,z方向的索引
            U: 位移向量
            nelx, nely, nelz: 网格在x,y,z方向的单元数量
        Returns:
            单元柔度贡献
        """
        # 计算节点编号
        n1 = (nely+1)*(nelz+1)*elx + (nelz+1)*ely + elz
        n2 = n1 + (nely+1)*(nelz+1)
        n3 = n2 + (nelz+1)
        n4 = n1 + (nelz+1)
        n5 = n1 + 1
        n6 = n2 + 1
        n7 = n3 + 1
        n8 = n4 + 1
        
        # 构建自由度索引
        edof = np.array([
            3*n1, 3*n1+1, 3*n1+2,
            3*n2, 3*n2+1, 3*n2+2,
            3*n3, 3*n3+1, 3*n3+2,
            3*n4, 3*n4+1, 3*n4+2,
            3*n5, 3*n5+1, 3*n5+2,
            3*n6, 3*n6+1, 3*n6+2,
            3*n7, 3*n7+1, 3*n7+2,
            3*n8, 3*n8+1, 3*n8+2
        ])
        
        # 提取单元位移
        Ue = U[edof]
        return float(Ue.T @ self.KE @ Ue)