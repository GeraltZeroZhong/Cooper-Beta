import numpy as np

class PCAAligner:
    """
    使用 PCA (主成分分析) 将点云数据的主轴对齐到 Z 轴。
    """
    def __init__(self):
        self.center = None
        self.rotation_matrix = None
        self.eigenvalues = None

    def fit(self, points):
        """
        计算点云的主轴。
        建议只传入 Beta-sheet 的 C-alpha 原子坐标以获得最佳效果。
        
        Args:
            points (np.ndarray): shape (N, 3) 的坐标数组
        """
        coords = np.array(points)
        if coords.shape[0] < 3:
            raise ValueError("点数太少，无法进行 PCA 分析")

        # 1. 计算中心并去中心化
        self.center = np.mean(coords, axis=0)
        centered_coords = coords - self.center

        # 2. 计算协方差矩阵
        # rowvar=False 表示每列是一个变量(x,y,z)，每行是一个观测
        cov_matrix = np.cov(centered_coords, rowvar=False)

        # 3. 特征值分解
        # eigh 适用于对称矩阵（协方差矩阵是对称的），且通常比 eig 更快更稳
        # 返回值 eig_vals 是升序排列的 (min -> max)
        # 返回值 eig_vecs 的每一列对应一个特征向量
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        self.eigenvalues = eig_vals
        self.rotation_matrix = eig_vecs
        
        # Log: 输出轴的方差比例，用于调试
        # variance_ratio = eig_vals / np.sum(eig_vals)
        # print(f"PCA Variance Ratios (X, Y, Z): {variance_ratio}")

    def transform(self, points):
        """
        将点云转换到新的 PCA 坐标系。
        转换后，Z 轴即为原点云的主轴方向。
        """
        if self.rotation_matrix is None:
            raise RuntimeError("Aligner is not fitted. Call fit() first.")
        
        coords = np.array(points)
        
        # 1. 平移（去中心化）
        centered = coords - self.center
        
        # 2. 旋转（投影到特征向量基）
        # result = centered @ eigenvectors
        # 结果的第0列对应最小特征值方向 (New X)
        # 结果的第1列对应中间特征值方向 (New Y)
        # 结果的第2列对应最大特征值方向 (New Z) -> 主轴
        transformed = np.dot(centered, self.rotation_matrix)
        
        return transformed

    def fit_transform(self, points):
        """fit 和 transform 的快捷组合"""
        self.fit(points)
        return self.transform(points)

# --- 单元测试 ---
if __name__ == "__main__":
    # 创建一个模拟的长条状圆柱体数据（模拟 Beta 桶）
    # 假设它沿着某个随机方向 (1, 1, 1) 延伸
    t = np.linspace(0, 20, 100)
    x = np.cos(t) + np.random.normal(0, 0.1, 100)
    y = np.sin(t) + np.random.normal(0, 0.1, 100)
    z = t * 2 # 长轴
    
    # 原始数据
    original_points = np.column_stack([x, y, z])
    
    # 随机旋转一下，使其不再对齐 Z 轴
    random_rot = np.array([[0.5, -0.866, 0], [0.866, 0.5, 0], [0, 0, 1]]) # 简单的 Z 轴旋转演示
    # 这里应该用更复杂的旋转，不过演示足够了
    
    aligner = PCAAligner()
    
    # 拟合
    transformed_points = aligner.fit_transform(original_points)
    
    print("原始数据 Z 轴方差:", np.var(original_points[:, 2]))
    print("对齐后数据 Z 轴方差:", np.var(transformed_points[:, 2]))
    print("对齐后数据 X 轴方差:", np.var(transformed_points[:, 0]))
    
    # 如果对齐成功，Transformed 的 Z 轴方差应该是最大的
    if np.var(transformed_points[:, 2]) > np.var(transformed_points[:, 0]):
        print("测试通过：主轴已成功对齐到 Z 轴。")
    else:
        print("测试失败：Z 轴不是最大方差方向。")